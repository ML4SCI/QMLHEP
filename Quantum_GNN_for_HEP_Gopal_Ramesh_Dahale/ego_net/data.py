from pathlib import Path
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.utils import k_hop_subgraph
import torch
import networkx as nx
from itertools import zip_longest
import numpy as np

def get_ego_dataset(dataset, n_hops):
	max_nodes = int(dataset.get_summary().num_nodes.max) # maximum number of nodes in the dataset
	k = n_hops

	# Prepare ego dataset
	temp_dataset = []
	max_ego_nodes = 0
	for data in dataset:
		n_nodes = data.num_nodes
		ego_nodes = []
		for node in range(n_nodes):
			try:
				subset, edge_index, _, _ = k_hop_subgraph(node,
													k,
													data.edge_index,
													directed=False)
			except:
				subset = []
				edge_index = torch.tensor([])
			n_subset_nodes = len(subset)

			if n_subset_nodes:
				G = nx.Graph()
				G.add_edges_from(edge_index.numpy().T)
				paths = nx.single_source_shortest_path_length(G, node, cutoff=k)

				nodes = np.array(list(paths.keys()))
				dists = np.array(list(paths.values()))

				hop_nodes = [
					[node] + list(nodes[np.where(dists == hop)[0]]) for hop in range(1, k + 1)
				]
				hop_nodes = np.array(list(zip_longest(*hop_nodes, fillvalue=max_nodes+1))).T
			else:
				dists = np.array([])
				hop_nodes = np.array([np.array([])]*k)
			ego_nodes.append(hop_nodes)
			max_ego_nodes = max(max_ego_nodes, hop_nodes.shape[-1])
		temp_dataset.append(ego_nodes)

	# Pad with maximum ego nodes
	ego_dataset = np.stack([np.stack([
		np.pad(ego_nodes, ((0, 0), (0, max_ego_nodes - ego_nodes.shape[-1])),
			   constant_values=max_nodes+1) for ego_nodes in data
	]) for data in temp_dataset]).astype(np.int32)

	return ego_dataset

# torch compatible dataset
class MUTAGDataset(Dataset):
	"""Mutag dataset."""
	def __init__(self, data_dir, n_hops=3):
		# Pad nodes and edges
		transform = T.Compose([T.Pad(28, 66)])
		self.tu_dataset = torch_geometric.datasets.TUDataset(root=data_dir, name="MUTAG", transform=transform)
		self.tu_dataset.shuffle()
		self.ego_dataset = get_ego_dataset(self.tu_dataset, n_hops)
		self.max_ego_nodes = self.ego_dataset.shape[3]

	def __len__(self):
		return len(self.tu_dataset)

	def __getitem__(self, idx):
		data = self.tu_dataset[idx]
		return {
			'x': data.x,
			'ego_graphs': torch.tensor(self.ego_dataset[idx]),
			'y': data.y
		}