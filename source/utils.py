import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

def rescale(array):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
    """
    array[array < 0] = 0 # Remove any under zero values
    #array = array/array.max() # Rescale to 0-1
    array = np.clip(array, 0, 255).astype(np.uint8) # Clip to 0,255 range
    
    return array


def saveimg(array: np.ndarray, name: str):
    """Saves ndarray as PIL image png.

    Args:
        Array (np.ndarray): Input image
        Name (string): Location to save
    """
    array = array.astype(np.uint8)
    array = Image.fromarray(array)
    array.save(name)


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def plot_curve(method_dict, filename):
        """
        Function for plotting the summed reward. May have issues
        if the total number of rewards is lower than 100.
        params:
        rewards: Tuple, tuple of summed reward over episode
        method: string, string to give the plot a proper title
        finished: boolean, bool to signify whether to save the plot y/n
        """
        plt.figure(figsize=(10, 6))

        for method, rewards_list in method_dict.items():
            # Calculate the mean and std without the nan values

            breakpoint()

            mean_rewards = np.mean(rewards_list, axis=1)
            std_rewards = np.std(rewards_list, axis=1)

            smoothmean = np.asarray(smooth(mean_rewards, 0.75))
            smoothstd = np.asarray(smooth(std_rewards, 0.75))

            # Determine the number of steps taken in the environment
            steps = np.arange(len(mean_rewards))
            # Plot the environment and create a std range of each method mean reward
            plt.plot(steps, smoothmean, label=method)
            plt.fill_between(steps,
                            smoothmean - smoothstd,
                            smoothmean + smoothstd,
                            alpha=0.1)

        plt.xlabel("Policy evaluation every 1000 steps")
        plt.ylabel("Summed average reward")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename}", dpi=300)