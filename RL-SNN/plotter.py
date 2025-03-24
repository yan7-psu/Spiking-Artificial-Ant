import matplotlib.pyplot as plt

# Enable interactive mode for live plotting
plt.ion()

def plot(scores, mean_scores, losses, mean_losses):
    # Clear the current figure to prepare for new plots
    plt.clf()

    # Get the current figure manager to manipulate the plot window
    manager = plt.get_current_fig_manager()
    manager.resize(1024, 768)
    manager.set_window_title("Santa Fe Training Plot")

    # Create the first subplot for scores
    plt.subplot(2, 1, 1)
    plt.title("Scores", fontsize=14)
    plt.xlabel("Number of Runs")
    plt.ylabel("Score")
    plt.plot(scores, label="Scores", color="blue")
    plt.plot(mean_scores, label="Mean Scores", color="orange")
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2f}")
    plt.legend()
    plt.grid()

    # Create the second subplot for loss
    plt.subplot(2, 1, 2)
    plt.title("Loss", fontsize=14)
    plt.xlabel("Number of Runs")
    plt.ylabel("Loss")
    plt.plot(losses, label="Loss", color="red")
    plt.plot(mean_losses, label="Mean Loss", color="purple")
    plt.text(len(losses) - 1, losses[-1], f"{losses[-1]:.4f}")
    plt.text(len(mean_losses) - 1, mean_losses[-1], f"{mean_losses[-1]:.4f}")
    plt.legend()
    plt.grid()

    # Adjust the layout to prevent overlapping of subplots and display the plot
    plt.tight_layout(h_pad=2)
    plt.show(block=False)
    plt.pause(0.1)
