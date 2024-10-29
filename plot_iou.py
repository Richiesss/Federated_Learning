import pandas as pd
import time
import matplotlib.pyplot as plt
import os


# Smoothing function as provided
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Function to update the plot in real-time
def update_plot():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    (line1,) = ax.plot(
        [], [], label="FedAvg"
    )  # Create empty plot lines for updating later
    (line2,) = ax.plot([], [], label="SOFA")

    while True:
        try:
            # Check if both CSV files exist
            if os.path.exists(
                "D:\workspace\scripts\Thesis_Research\FL\FL-sys\SOFA_metrics.csv"
            ) and os.path.exists(
                "D:\workspace\scripts\Thesis_Research\FL\FL-sys\SOFA_metrics.csv"
            ):
                # Reload the CSV files
                df1_iid = pd.read_csv(
                    "D:\workspace\scripts\Thesis_Research\FL\FL-sys\SOFA_metrics.csv"
                )
                df2_sofa_iid = pd.read_csv(
                    "D:\workspace\scripts\Thesis_Research\FL\FL-sys\SOFA_metrics.csv"
                )

                # Apply the smoothing function to the 'Accuracy' column
                df1_iid["IoU"] = smooth_curve(df1_iid["IoU"], factor=0.8)
                df2_sofa_iid["IoU"] = smooth_curve(df2_sofa_iid["IoU"], factor=0.8)

                # Update the data for the lines
                line1.set_data(df1_iid["Round"], df1_iid["IoU"])
                line2.set_data(df2_sofa_iid["Round"], df2_sofa_iid["IoU"])

                # Adjust x and y axis limits dynamically
                ax.set_xlim(0, max(df1_iid["Round"].max(), df2_sofa_iid["Round"].max()))
                ax.set_ylim(0, 1)  # Accuracy values should be between 0 and 1

                # Adding titles and labels
                ax.set_title(f"IoU Per Round: FedAvg vs SOFA (non-IID, {time.ctime()})")
                ax.set_xlabel("Round")
                ax.set_ylabel("IoU")
                ax.legend()
                ax.grid(True)

                # Redraw the updated plot
                fig.canvas.draw()

        except Exception as e:
            print(f"An error occurred: {e}")

        # Pause to update the plot every 5 seconds
        plt.pause(5)


# Call the update_plot function to start real-time plotting
update_plot()
