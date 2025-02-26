import plotly.express as px
import plotly.subplots as sp


def generate_visual_insights(df):
    """
    Generates various visualizations based on emotion and topic data.

    Args:
        df (pd.DataFrame): The dataset containing emotion and topic information.
    """
    print('start generate_visual_insights')
    # Drop non-emotion columns to work only with emotion scores
    emotion_columns = [col for col in df.columns if
                       col not in ['document_id', 'Topic_number', 'year', 'Average_Emotions_Score']]
    dominant_emotions = df[emotion_columns].idxmax(axis=1).value_counts()
    average_emotion_per_year = df.groupby('year')['Average_Emotions_Score'].mean()
    emotion_distribution_by_topic = df.groupby('Topic_number')[emotion_columns].mean()
    top_emotional_docs = df[df['Topic_number'] != -1].nlargest(30, 'Average_Emotions_Score')

    # Filter documents to include only those with Topic_number between 0 and 20
    filtered_documents = df[df["Topic_number"].between(0, 20)]

    # Function to generate the graph for a selected topic
    def visualize_topic_trends(topic_number):
        filted_df = df[df["Topic_number"] == topic_number]
        #dominant_emotions = filted_df[emotion_columns].idxmax(axis=1).value_counts()

        # Prepare the data for visualization
        melted_df = filted_df.melt(
            id_vars=["year", "Topic_number"],
            value_vars=dominant_emotions.index,
            var_name="Emotion",
            value_name="Score"
        )

        # Group by year and emotion, then calculate the average score
        avg_emotion_scores = melted_df.groupby(["year", "Emotion"], as_index=False)["Score"].mean()

        # Calculate the overall average score for each emotion across all years
        overall_avg_scores = avg_emotion_scores.groupby("Emotion", as_index=False)["Score"].mean()

        # Sort emotions by their overall average score and select the top 10
        top_10_emotions = overall_avg_scores.nlargest(10, "Score")["Emotion"]

        # Filter the average emotion scores to include only the top 10 emotions
        top_10_avg_scores = avg_emotion_scores[avg_emotion_scores["Emotion"].isin(top_10_emotions)]

        # Create the line plot for average emotion scores over time
        fig = px.line(
            top_10_avg_scores,
            x="year",
            y="Score",
            color="Emotion",
            title=f"Average Emotion Score Over Time for Topic {topic_number} (Top 10 Emotions)",
            labels={"year": "Year", "Score": "Average Emotion Score", "Emotion": "Emotion"},
            height=400
        )
        return fig

    def plot_two_topics():
        # Create subplots with 2 rows and 1 column
        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Topic 1", "Topic 2"))

        # Generate the plot for Topic 1
        fig1 = visualize_topic_trends(1)
        if fig1:
            for trace in fig1.data:
                fig.add_trace(trace, row=1, col=1)

        # Generate the plot for Topic 2
        fig2 = visualize_topic_trends(2)
        if fig2:
            for trace in fig2.data:
                fig.add_trace(trace, row=2, col=1)

        # Update layout for better spacing and titles
        fig.update_layout(
            height=800,  # Adjust height to accommodate both plots
            showlegend=True,
            title_text="Average Emotion Scores Over Time for Topics 1 and 2 (Top 10 Emotions)"
        )

        # Show the combined plot
        fig.show()

    # Call the function to display both graphs
    plot_two_topics()

"""

    # Create an interactive line plot for trends over time
    fig_trends_overtime = px.line(
        melted_df,
        x="year",
        y="Score",
        color="Emotion",
        facet_col="Topic_number",
        title="Trends Over Time by Emotion and Topic (0–20)",
        labels={"year": "Year", "Score": "Emotion Score", "Topic_number": "Topic"},
        height=800
    )
    fig_trends_overtime.show()
"""

"""
    # 1. Dominant Emotions by Document
    fig_dominant_emotions = px.bar(
        dominant_emotions,
        title="Dominant Emotions by Document",
        labels={"index": "Emotion", "value": "Count"},
        text=dominant_emotions.values,
    )
    fig_dominant_emotions.update_traces(texttemplate='%{text}', textposition='outside')
    fig_dominant_emotions.show()

    # 2. Trends Over Time
    fig_trends_over_time = px.line(
        average_emotion_per_year,
        title="Trends Over Time: Average Emotion Scores by Year",
        labels={"year": "Year", "value": "Average Emotion Score"},
    )
    fig_trends_over_time.show()

    # 3. Emotion Distribution by Topic
    emotion_distribution_by_topic.reset_index(inplace=True)
    fig_emotion_distribution = px.bar(
        emotion_distribution_by_topic.melt(
            id_vars=["Topic_number"],
            var_name="Emotion",
            value_name="Score"
        ),
        x="Topic_number",
        y="Score",
        color="Emotion",
        title="Emotion Distribution by Topic",
        barmode="group",
    )
    fig_emotion_distribution.show()


    # 4. Highly Emotional Documents
    fig_highlight_emotional_docs = px.bar(
        top_emotional_docs.melt(id_vars=['document_id'], var_name='', value_name='Score'),
        x="document_id",
        y="Score",
        title="Top 30 Highly Emotional Documents",
        labels={"document_id": "Document ID", "Average_Emotions_Score": "Average Emotion Score"},
    )

    # Display visualizations
    fig_dominant_emotions.show()
    fig_trends_over_time.show()
    fig_emotion_distribution.show()
    fig_highlight_emotional_docs.show()

"""
