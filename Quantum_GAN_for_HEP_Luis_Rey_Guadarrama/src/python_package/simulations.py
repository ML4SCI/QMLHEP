import numpy as np
import matplotlib.pyplot as plt

class MonteCarlo:
    """
    A class to simulate various probabilistic experiments such as rolling dice, coin tosses,
    particle decay, and random walks.

    Methods:
    -------
    rolling_dice(n_samples: int) -> np.array:
        Simulate rolling two six-sided dice and return the sum of the results.
    
    coin_toss(n_samples: int, coins: int) -> np.array:
        Simulate tossing a specified number of coins and return the number of heads for each set of tosses.
    
    particle_decay(n_particles: int, decay_constant: float) -> np.array:
        Simulate the decay times of particles based on an exponential distribution.
    
    random_walk(samples: int, steps: int) -> np.array:
        Simulate a 2D random walk for a given number of samples and steps.
    """
    
    @staticmethod
    def rolling_dice(n_samples: int) -> np.array:
        """
        Simulate rolling two six-sided dice and return the sum of the results.

        Parameters:
        ----------
        n_samples : int
            The number of times to roll the dice.

        Returns:
        -------
        np.array
            An array containing the sum of the results of rolling two six-sided dice
            `n_samples` times.
        """
        dice_1 = np.random.randint(1, 7, n_samples)
        dice_2 = np.random.randint(1, 7, n_samples)
        dice_sum = dice_1 + dice_2

        return dice_sum
    

    @staticmethod
    def coin_toss(n_samples: int, coins: int) -> np.array:
        """
        Simulate tossing a specified number of coins and return the number of heads for each set of tosses.

        *Parameters:*
        ----------
        n_samples : int
            The number of sets of coin tosses to simulate.
        coins : int
            The number of coins to toss in each set.

        *Returns:*
        -------
        np.array
            An array containing the number of heads (1's) in each set of coin tosses.
        """
        coin_tosses = np.random.randint(2, size=(n_samples, coins))
        return np.sum(coin_tosses, axis=1)
    

    @staticmethod
    def particle_decay(n_particles: int, decay_constant: float) -> np.array:
        """
        Simulate the decay times of particles based on an exponential distribution.

        *Parameters:*
        ----------
        n_particles : int
            The number of particles to simulate.
        decay_constant : float
            The decay constant (lambda) for the exponential distribution.

        *Returns:*
        -------
        np.array
            An array containing the decay times of the particles, rounded to the nearest integer.
        """

        decay_time = np.random.exponential(scale=1/decay_constant, size=n_particles)
        decay_time = np.round(decay_time - 0.5)
        return decay_time
    
    
    @staticmethod
    def random_walk(n_samples: int, n_steps: int) -> np.array:
        """
        Simulate a 2D random walk for a given number of samples and steps. It takes step of length 1.0 in a random direction.

        *Parameters:*
        ----------
        n_samples : int
            The number of random walk paths to simulate.
        n_steps : int
            The number of steps for each random walk path.

        *Returns:*
        -------
        np.array
            An array of shape (samples, 2) containing the final positions of each random walk.
        """
        positions = np.zeros(shape=(n_samples, 2))

        for step in range(n_steps):
            step = np.random.random(size=(n_samples, 2)) - 0.5
            norms = np.linalg.norm(step, axis=1)
            normalized_step = step/norms[:, np.newaxis]
            positions += normalized_step

        return positions
    
    @staticmethod
    def probability_distribution_1D(samples: np.array, batch_size: int, bins: int) -> np.array:
        probs = []
        
        for i in range(int(len(samples)/batch_size)):
            counts = np.zeros(bins)
            data = samples[i*batch_size:(i+1)*batch_size]
            # Count the occurrences of each result
            for value in data:
                counts[int(value)] += 1
            
            # Calculate probabilities by normalizing the counts
            probabilities = counts / len(data)
            probs.append(probabilities)
        
        return np.array(probs)
    
    @staticmethod
    def probability_distribution_2D(samples: np.array, batch_size: int, bins: int, limits: tuple[int, int]) -> np.array:
        x_bins = np.linspace(limits[0], limits[1], bins + 1)
        y_bins = np.linspace(limits[0], limits[1], bins + 1)
        t_samples = samples.T
        n_batches = int(np.size(t_samples, axis=1)/batch_size)
        counts_map = []

        for i in range(n_batches):
            data = t_samples[:,i*batch_size:(i+1)*batch_size]
            map, _ , _ = np.histogram2d(data[0], data[1], bins=(x_bins, y_bins))
            counts_map.append(map)
        
        prob_map = np.array(counts_map)/batch_size

        return prob_map
    
    
    @staticmethod
    def visualize_distribution_1D(samples: np.array, bins: int, xlabel: str = "samples") -> None:
        plt.hist(samples, bins=bins, align='left', range=[0, bins], rwidth=0.8, density=True, color="cornflowerblue")
        plt.xlabel(xlabel)
        plt.ylabel("pseudo-probabilities")
        plt.title("Probability Distribution")
        plt.show()


    @staticmethod
    def visualize_distribution_2D(samples: np.array, limits: tuple, bins: int, xlabel:str = "X", ylabel:str = "Y", vmin:float = 0, vmax:float = 0.5) -> None:

        batch_size = np.size(samples, axis=0)
        probs = MonteCarlo.probability_distribution_2D(samples, batch_size, bins, limits)[0]
        plt.figure(figsize=(8, 6))
        plt.imshow(probs, extent=[limits[0], limits[1], limits[0], limits[1]], origin='lower', cmap='inferno', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Density')
        plt.title('Probability Distribution')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()