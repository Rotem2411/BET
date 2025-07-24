import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.manifold import TSNE
import umap
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

def generate_visual_insights(df):
    print('start generate_visual_insights')
    # Drop non-emotion columns to work only with emotion scores
    emotion_columns = [col for col in df.columns if
                       col not in ['document_id', 'Topic_number', 'year', 'Average_Emotions_Score']]
    dominant_emotions = df[emotion_columns].idxmax(axis=1).value_counts()

    # Function to generate the graph for a selected topic
    def visualize_topic_trends(topic_number):
        filted_df = df[df["Topic_number"] == topic_number]

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

def plot_metrics(metrics_df, x_col, y_cols_labels_colors, title):
    plt.figure(figsize=(6 * len(y_cols_labels_colors), 5))
    plt.suptitle(title, fontsize=16)
    for i, (y_col, label, color) in enumerate(y_cols_labels_colors, 1):
        plt.subplot(1, len(y_cols_labels_colors), i)
        plt.plot(metrics_df[x_col], metrics_df[y_col], marker='o', color=color)
        plt.title(f"{label} vs. {x_col}")
        plt.xlabel(x_col)
        plt.ylabel(label)
        plt.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_embedding_clusters(df, embeddings, method='umap', n_components=3):
    print('start visualize_embedding_clusters')
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        embeddings_reduced = reducer.fit_transform(embeddings)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
        embeddings_reduced = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Method must be 'umap' or 'tsne'.")

    # Create a new DataFrame with reduced dimensions
    df_reduced = pd.DataFrame({
        'category_index': df['category_index'],
        'word_index': df['word_index'],
        'word': df['word'],
        'x': embeddings_reduced[:, 0],
        'y': embeddings_reduced[:, 1],
        'z': embeddings_reduced[:, 2] if n_components == 3 else None
    })

    # Plot the embeddings using Plotly Express
    if n_components == 3:
        fig = px.scatter_3d(df_reduced, x='x', y='y', z='z',
                            color='category_index', hover_data=['word'],
                            title=f'Word Embeddings ({method.upper()} 3D)',
                            width=1200, height=900)
    else:
        fig = px.scatter(df_reduced, x='x', y='y',
                         color='category_index', hover_data=['word'],
                         title=f'Word Embeddings ({method.upper()} 2D)')

    output_file = os.path.join(OUTPUT_DIR, f"word_embeddings_{method}_{n_components}D.html")
    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")

def visualize_performance_models(file_path="summary_evaluation_metrics.csv", save_figures=True):
    df = pd.read_csv(file_path)
    df["num_outliers"] = pd.to_numeric(df["num_outliers"], errors="coerce")
    df["distance_threshold"] = pd.to_numeric(df["distance_threshold"], errors="coerce")
    df["nr_topics"] = pd.to_numeric(df["nr_topics"], errors="coerce")
    topic_models = df["Topic Model"].unique()

    for topic_model in topic_models:
        model_df = df[df["Topic Model"] == topic_model].copy()
        model_df["Clustering Model"] = model_df["Clustering Model"].fillna("")
        model_df["Model_Name"] = model_df.apply(
            lambda row: f"{row['Topic Model']}_{row['sentence_model']}" if row["Topic Model"] == "LDA"
            else f"{row['Topic Model']}_{row['Clustering Model']}_{row['sentence_model']}", axis=1)

        unique_models = model_df["Model_Name"].unique()
        for model_name in unique_models:
            variant_df = model_df[model_df["Model_Name"] == model_name]
            clustering_model = variant_df["Clustering Model"].iloc[0]
            sentence_model = variant_df["sentence_model"].iloc[0]

            # Determine x-axis and secondary metric
            if clustering_model == "AgglomerativeClustering":
                x_axis = "distance_threshold"
                secondary_metric = "nr_topics"
                secondary_label = "Number of Topics"
                secondary_color = "purple"
            else:
                x_axis = "nr_topics"
                secondary_metric = "num_outliers" if clustering_model.lower() == "hdbscan" else None
                secondary_label = "Number of Outliers"
                secondary_color = "red"

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.set_title(f"{model_name}")
            sns.lineplot(data=variant_df, x=x_axis, y="coherence_score", label="Coherence Score", ax=ax1)
            sns.lineplot(data=variant_df, x=x_axis, y="topic_diversity", label="Topic Diversity", ax=ax1, color="green")
            ax1.set_xlabel("Distance Threshold" if x_axis == "distance_threshold" else "Number of Topics")
            ax1.set_ylabel("Score")
            ax1.grid(True)

            # Secondary y-axis
            if secondary_metric:
                ax2 = ax1.twinx()
                sns.lineplot(data=variant_df, x=x_axis, y=secondary_metric, ax=ax2, label=secondary_label, color=secondary_color)
                ax2.set_ylabel(secondary_label)
                ax2.legend(loc='upper right')
            ax1.legend(loc='upper left')

            plt.tight_layout()
            if save_figures:
                safe_name = model_name.replace("/", "_")
                plt.savefig(f"{OUTPUT_DIR}/{safe_name}_metrics.png", dpi=300)
            plt.show()

        # Combined comparison plot(s)
        if topic_model == "LDA":
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f"{topic_model} Model: Comparison", fontsize=18)

            sns.lineplot(data=model_df, x="nr_topics", y="coherence_score", hue="sentence_model", ax=axes[0])
            axes[0].set_title("Coherence Score")
            axes[0].set_xlabel("Number of Topics")
            axes[0].set_ylabel("Coherence Score")
            axes[0].grid(True)

            sns.lineplot(data=model_df, x="nr_topics", y="topic_diversity", hue="sentence_model", ax=axes[1], palette="Set2")
            axes[1].set_title("Topic Diversity")
            axes[1].set_xlabel("Number of Topics")
            axes[1].set_ylabel("Topic Diversity")
            axes[1].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_figures:
                plt.savefig(f"{OUTPUT_DIR}/{topic_model}_comparison.png", dpi=300)
            plt.show()

        else:
            for cluster_type in ["HDBSCAN", "AgglomerativeClustering"]:
                sub_df = model_df[model_df["Clustering Model"] == cluster_type]
                if sub_df.empty:
                    continue
                x_var = "distance_threshold" if cluster_type == "AgglomerativeClustering" else "nr_topics"

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle(f"{topic_model} ({cluster_type}): Comparison", fontsize=18)

                sns.lineplot(data=sub_df, x=x_var, y="coherence_score", hue="sentence_model", ax=axes[0])
                axes[0].set_title("Coherence Score")
                axes[0].set_xlabel("Distance Threshold" if x_var == "distance_threshold" else "Number of Topics")
                axes[0].set_ylabel("Coherence Score")
                axes[0].grid(True)

                sns.lineplot(data=sub_df, x=x_var, y="topic_diversity", hue="sentence_model", ax=axes[1], palette="Set2")
                axes[1].set_title("Topic Diversity")
                axes[1].set_xlabel("Distance Threshold" if x_var == "distance_threshold" else "Number of Topics")
                axes[1].set_ylabel("Topic Diversity")
                axes[1].grid(True)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                if save_figures:
                    plt.savefig(f"{OUTPUT_DIR}/{topic_model}_{cluster_type}_comparison.png", dpi=300)
                plt.show()