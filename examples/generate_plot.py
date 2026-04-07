import pandas as pd
import plotly.express as px

# Load the data
file_path = 'lambda-cloud-api/availability_stats.csv'
df = pd.read_csv(file_path)

# Convert timestamp to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter out rows where type is not needed or data is messy if necessary
# For now, we plot availability status over time per GPU type
fig = px.scatter(df, x='timestamp', y='type', color='available',
                 title='GPU Availability Over Time',
                 labels={'available': 'Is Available', 'timestamp': 'Time', 'type': 'GPU Model'},
                 hover_data=['region_count', 'regions'])

# Improve layout
fig.update_layout(xaxis_title='Timestamp', yaxis_title='GPU Model')
fig.show()

# Save the plot to an HTML file
fig.write_html('gpu_availability.html')
print("Plot saved to gpu_availability.html")
