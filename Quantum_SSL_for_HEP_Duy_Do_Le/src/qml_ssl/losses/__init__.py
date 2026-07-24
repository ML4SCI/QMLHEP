import torch
import torch.nn.functional as F

def check_shapes(embeddings, labels):
    if labels is not None and embeddings.shape[0] != labels.shape[0]:
        raise ValueError("Number of embeddings must equal number of labels")
    if labels is not None and labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x

def set_ref_emb(embeddings, labels, ref_emb, ref_labels):
    if ref_emb is not None:
        if ref_labels is not None:
            ref_labels = to_device(ref_labels, ref_emb)
    else:
        ref_emb, ref_labels = embeddings, labels
    check_shapes(ref_emb, ref_labels)
    return ref_emb, ref_labels

def pos_pairs_from_tuple(indices_tuple):
    return indices_tuple[:2]


def neg_pairs_from_tuple(indices_tuple):
    return indices_tuple[2:]

def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def neg_inf(dtype):
    return torch.finfo(dtype).min

def small_val(dtype):
    return torch.finfo(dtype).tiny

def labels_or_indices_tuple_required(labels, indices_tuple):
    if labels is None and indices_tuple is None:
        raise ValueError("labels and indices_tuple cannot both be None")

def get_matches_and_diffs(labels, ref_labels=None):
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    return matches, diffs

def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    matches, diffs = get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx

def convert_to_pairs(indices_tuple, labels, ref_labels=None):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels, ref_labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n

def alignment_loss(embeddings1, embeddings2):
    # Alignment loss over the batch
    return torch.mean(torch.norm(embeddings1 - embeddings2, dim=1) ** 2)

def uniformity_loss(embeddings, t=2):
    # Normalize embeddings to lie on the unit hypersphere
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Compute pairwise distances
    distances = torch.cdist(embeddings, embeddings, p=2) ** 2
    # Apply the uniformity loss formula
    uniformity = torch.mean(torch.exp(-t * distances))
    return uniformity

class BaseMetricLossFunction(torch.nn.Module):
    """
    Base class for metric learning losses. This class defines the main flow of 
    how embeddings are processed and the loss is computed, and it includes 
    utility functions for regularization, embedding checks, and loss computation.
    """

    def __init__(self, distance=None, reducer=None, mat_based_loss=False, **kwargs):
        super().__init__(**kwargs)
        self.distance = self.get_default_distance() if distance is None else distance
        # self.reducer = self.get_default_reducer() if reducer is None else reducer
        self.set_reducer(reducer)
        self.loss_method = (
            self.mat_based_loss if mat_based_loss else self.pair_based_loss
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = convert_to_pairs(indices_tuple, labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        # print(mat)
        return self.loss_method(mat, indices_tuple)

    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None
    ):
        """
        Main forward method which computes the loss. Handles the embedding regularization,
        embedding checks, and passes the relevant information to compute the loss.

        Args:
            embeddings: tensor of shape (batch_size, embedding_size)
            labels: tensor of shape (batch_size) (optional)
            indices_tuple: tuple containing anchor, positive, and negative indices 
                           to define the pairings for the loss (optional)

        Returns:
            The computed loss as a dictionary containing loss values, indices, and 
            reduction type.
        """
        self.reset_stats()
        check_shapes(embeddings, labels)
        if labels is not None:
            labels = to_device(labels, embeddings)
        ref_emb, ref_labels = set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels
        )
        # print(loss_dict, embeddings, labels)
        # self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
    
    def zero_loss(self):
        """ Return a default zero loss structure. """
        return {"losses": 0, "indices": None, "reduction_type": "already_reduced"}

    def zero_losses(self):
        """ Return a dictionary of zero losses for all sub-loss names. """
        return {loss_name: self.zero_loss() for loss_name in self.sub_loss_names()}

    def _sub_loss_names(self):
        """ Define the sub-loss names that derived classes should implement. """
        return ["loss"]

    def pair_based_loss(self, mat, indices_tuple):
        """
        Computes pair-based loss for metric learning. This function computes the 
        distances between positive and negative pairs given a distance matrix.

        Args:
            mat: distance matrix between embeddings.
            indices_tuple: tuple containing indices for anchor, positive, and negative pairs.
        
        Returns:
            A dictionary containing the positive and negative pair losses.
        """
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        return self._compute_loss(pos_pair, neg_pair, indices_tuple)
    
    def mat_based_loss(self, mat, indices_tuple):
        a1, p, a2, n = indices_tuple
        pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
        pos_mask[a1, p] = 1
        neg_mask[a2, n] = 1
        return self._compute_loss(mat, pos_mask, neg_mask)

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        """
        Computes the final loss given positive and negative pair distances.
        Derived classes should override this method to implement the specific 
        loss function (e.g., contrastive loss, triplet loss).
        """
        raise NotImplementedError

    def get_default_distance(self):
        return DotProductSimilarity()

    def get_default_reducer(self):
        return MeanReducer()

    def set_reducer(self, reducer):
        import copy
        if isinstance(reducer, MultipleReducers):
            self.reducer = reducer
        elif len(self.sub_loss_names()) == 1:
            self.reducer = (
                self.get_default_reducer()
                if reducer is None
                else copy.deepcopy(reducer)
            )
        else:
            reducer_dict = {}
            for k in self.sub_loss_names():
                reducer_dict[k] = (
                    self.get_default_reducer()
                    if reducer is None
                    else copy.deepcopy(reducer)
                )
            self.reducer = MultipleReducers(reducer_dict)

    def sub_loss_names(self):
        return ["loss"]

    def reset_stats(self):
        for attr_list in ["_record_these_stats"]:
            for r in getattr(self, attr_list, []):
                setattr(self, r, 0)

class ContrastiveLoss(BaseMetricLossFunction):
    """
    Implementation of the contrastive loss function, where the goal is to minimize 
    the distance between positive pairs (with a margin of 0) and maximize the distance 
    between negative pairs (with a margin > 0).
    """

    def __init__(self, pos_margin=0, neg_margin=1, **kwargs):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )

    def _compute_loss(self, pos_pair_dist, neg_pair_dist, indices_tuple):
        """
        Computes contrastive loss given the distances between positive and negative pairs.

        Args:
            pos_pair_dist: distance between positive pairs.
            neg_pair_dist: distance between negative pairs.
            indices_tuple: tuple containing the indices of anchors, positives, and negatives.

        Returns:
            A dictionary containing positive and negative losses.
        """
        pos_loss, neg_loss = 0, 0
        if len(pos_pair_dist) > 0:
            pos_loss = self.get_per_pair_loss(pos_pair_dist, "pos")
        if len(neg_pair_dist) > 0:
            neg_loss = self.get_per_pair_loss(neg_pair_dist, "neg")
        pos_pairs = pos_pairs_from_tuple(indices_tuple)
        neg_pairs = neg_pairs_from_tuple(indices_tuple)
        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair",
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair",
            },
        }

    def get_per_pair_loss(self, pair_dists, pos_or_neg):
        """
        Calculates the loss per pair for either positive or negative pairs.

        Args:
            pair_dists: distances between pairs.
            pos_or_neg: string indicating whether to compute for positive or negative pairs.

        Returns:
            The computed loss for the given pairs.
        """
        loss_calc_func = self.pos_calc if pos_or_neg == "pos" else self.neg_calc
        margin = self.pos_margin if pos_or_neg == "pos" else self.neg_margin
        per_pair_loss = loss_calc_func(pair_dists, margin)
        return per_pair_loss

    def pos_calc(self, pos_pair_dist, margin):
        """ Computes the positive pair loss with margin. """
        return torch.nn.functional.relu(self.distance.margin(pos_pair_dist, margin))

    def neg_calc(self, neg_pair_dist, margin):
        """ Computes the negative pair loss with margin. """
        return torch.nn.functional.relu(self.distance.margin(margin, neg_pair_dist))

    def _sub_loss_names(self):
        """ Define the sub-loss names specific to contrastive loss. """
        return ["pos_loss", "neg_loss"]

class NTXentLoss(BaseMetricLossFunction):
    """
    Implementation of the NT-Xent (Normalized Temperature-scaled Cross Entropy Loss),
    commonly used in self-supervised contrastive learning.
    """

    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        # self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        """
        Computes the NT-Xent loss given the positive and negative pairs.

        Args:
            pos_pairs: positive pair distances.
            neg_pairs: negative pair distances.
            indices_tuple: tuple containing indices of anchor, positive, and negative pairs.

        Returns:
            A dictionary containing the computed NT-Xent loss.
        """
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()


class DotProductSimilarity(torch.nn.Module):
    def __init__(self, is_inverted=True, power=1):
        super().__init__()
        self.power = power
        self.is_inverted = is_inverted

    def compute_mat(self, query_emb, ref_emb):
        return torch.matmul(query_emb, ref_emb.t())

    def pairwise_distance(self, query_emb, ref_emb):
        return torch.sum(query_emb * ref_emb, dim=1)
    
    def normalize(self, embeddings, p=2, dim=1, **kwargs):
        return torch.nn.functional.normalize(embeddings, p=p, dim=dim, **kwargs)
    
    def forward(self, query_emb, ref_emb=None):
        query_emb_normalized = self.normalize(query_emb)  
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.normalize(ref_emb)
        # self.set_default_stats(
        #     query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        # )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat**self.power
        assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        return mat

 
class BaseReducer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.add_to_recordable_attributes(name="losses_size", is_stat=True)

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        # self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        # self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError
    
    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)

    def zero_loss(self, embeddings):
        return torch.sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        if (not torch.is_tensor(losses)) and (losses == 0):
            return True
        return False

    def reset_stats(self):
        for attr_list in ["_record_these_stats"]:
            for r in getattr(self, attr_list, []):
                setattr(self, r, 0)

class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        return torch.mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

class MultipleReducers(BaseReducer):
    def __init__(self, reducers, default_reducer=None, **kwargs):
        super().__init__(**kwargs)
        self.reducers = torch.nn.ModuleDict(reducers)
        self.default_reducer = (
            MeanReducer() if default_reducer is None else default_reducer
        )

    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        sub_losses = torch.zeros(
            len(loss_dict), dtype=embeddings.dtype, device=embeddings.device
        )
        loss_count = 0
        for loss_name, loss_info in loss_dict.items():
            input_dict = {loss_name: loss_info}
            if loss_name in self.reducers:
                loss_val = self.reducers[loss_name](input_dict, embeddings, labels)
            else:
                loss_val = self.default_reducer(input_dict, embeddings, labels)
            sub_losses[loss_count] = loss_val
            loss_count += 1
        return self.sub_loss_reduction(sub_losses, embeddings, labels)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        return torch.sum(sub_losses)
   