from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Remove the size limit

import streamlit as st
import pickle
import numpy as np

st.title("英検CSEスコア予測アプリ")
st.markdown("""
英検CSEスコア予測ツールへようこそ!
このアプリは、校内試験の結果に基づいて英検CSEスコアを推定し、効果的に準備するための個別のフィードバックを提供します。
""")

# Function to load a model
def load_model(file_name):
    """
    Load the trained model from a file.
    """
    try:
        with open(file_name, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {file_name}")
        return None

# Function to make a prediction
def predict(model, exam_score):
    """
    Use the trained model to predict the Eiken CSE score.
    """
    prediction = model.predict(np.array([[exam_score]]))[0]
    return round(prediction)

# Define passing scores for each Eiken level
eiken_levels = {
    "3級": 1103,
    "準2級": 1322,
    "2級": 1520
}

# User input for grade level and exam type
grade = st.selectbox("学年", [1, 2, 3])  # Add more grades if needed
exam_type = st.selectbox("定期試験の種類", ["1学期中間","1学期期末", "2学期期末"])
exam_score = st.slider("定期試験得点を教えてください！", min_value=0, max_value=100, value=50)

# User input for Eiken level
eiken_level = st.selectbox("英検受験級", list(eiken_levels.keys()))
passing_score = eiken_levels[eiken_level]

if st.button("予測"):
    # Determine model file based on grade and exam type
    model_file = f"models/grade{grade}_{exam_type.lower().replace(' ', '_')}_model.pkl"
    model = load_model(model_file)

    if model:
    # Make prediction
        predicted_cse = predict(model, exam_score)

        # Get the contextual Eiken round text
        eiken_round_text = {
            "1学期中間": "Eiken 第1回",
            "1学期期末": "Eiken 第2回",
            "2学期期末": "Eiken 第3回"
        }
        eiken_round = eiken_round_text[exam_type]

        # Display results in a styled card
        st.markdown(
            f"""
            <div style="
                background-color: #fffde7; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); 
                margin-bottom: 20px;
            ">
                <h2 style="color: #1a73e8; font-size: 32px; text-align: center;">
                    予測結果
                </h2>
                <p style="font-size: 22px; color: #000000; text-align: center;">
                    {eiken_round}の予測CSEスコア: <strong>{predicted_cse}</strong>
                </p>
                <p style="font-size: 18px; color: #555555; text-align: center;">
                    {eiken_level}の合格スコア: <strong>{passing_score}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Add feedback outside the card for clarity
        if predicted_cse >= passing_score:
            st.success("🎉 おめでとうございます！予測スコアが合格点に達している、またはそれを上回っています！")
            if predicted_cse - passing_score <= 100:
                st.warning(
                    "⚠️⚠️あなたの予測スコアは合格点の100点以内に収まっています。"
                    "これは、現在の英語力がこの級の英検合格に必要なレベルにぎりぎり到達していることを示しています。"
                    "実際のスコアが予測スコアに見合うようにするため、英検対策をしっかり行うことを強くお勧めします！⚠️⚠️"
                )
        elif predicted_cse < passing_score:
            st.error("🚨 予測スコアが合格点を下回っています。")
            if passing_score - predicted_cse <= 100:
                st.warning(
                    "❗予測スコアは合格点を下回っていますが、100点未満の差です。"
                    "これは、現在の英語力が合格レベルにとても近いことを意味します！ "
                    "英検対策を少し行えば、合格できる可能性が高いと思います！"
                    "そのため、英語の先生にどのように準備すればよいか相談してみてください！❗"
                )


        st.progress(min(predicted_cse / passing_score, 1.0))

        import streamlit as st
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import rcParams

        # Configure font for Japanese support
        rcParams['font.family'] = 'YuGothic'

        # Data
        x_labels = ["予測スコア", "合格スコア"]
        y_values = [predicted_cse, passing_score]
        bar_width = 0.5  # Bar width

        # Adjust x-axis positions for the bars
        x_positions = np.arange(len(x_labels))  # [0, 1] for two bars

        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 4))  # Adjust figure size if needed
        ax.bar(x_positions, y_values, color=["blue", "green"], width=bar_width)

        # Add a horizontal line from the top of the predicted score bar
        predicted_x = x_positions[0]  # X-position of the predicted score bar
        predicted_y = y_values[0]  # Y-value of the predicted score bar (height)
        ax.hlines(predicted_y, predicted_x - bar_width / 2, x_positions[1] + bar_width / 2, colors="red", linestyles="dashed", linewidth=1.5)

        # Add a comment if the predicted score is 100 points or less above the passing score
        if predicted_cse - passing_score <= 100:
            ax.annotate(
                "ギリギリ！",  # Annotation text
                xy=(x_positions[1], predicted_y),  # Position of the annotation (near the line)
                xytext=(x_positions[1] + 0.2, predicted_y + 20),  # Adjust the text position
                fontsize=12,
                color="red",
                arrowprops=dict(arrowstyle="->", color="red", linewidth=1.5)  # Optional arrow pointing to the line
            )

        # Customize the x-axis
        ax.set_xticks(x_positions)  # Set x-tick positions
        ax.set_xticklabels(x_labels, fontsize=12)  # Set x-tick labels
        ax.set_xlim(-0.5, len(x_labels) - 0.5)  # Ensure the bars are centered
        ax.set_ylabel("スコア", fontsize=12)
        ax.set_title(f"{eiken_level}のスコア分析", fontsize=14)

        # Convert the figure to a PNG image
        from io import BytesIO
        import base64

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        st.markdown(
            f"""
            <div style="
                background-color: #ffffff; 
                padding: 20px; 
                border-radius: 10px; 
                border: 2px solid #1a73e8; 
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.7); 
                margin: 20px 0;
                text-align: center;
            ">
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 600px; border-radius: 10px;">
            </div>
            """,
            unsafe_allow_html=True,
        )
