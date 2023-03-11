import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

st.sidebar.title("Whatsapp Chat Analyzer")

try:
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        # Fetching unique user
        user_list = df['user'].unique().tolist()

        if 'group_notification' in user_list:
            user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")
        selected_user = st.sidebar.selectbox("Show Analysis wrt", user_list)

        if st.sidebar.button("Show Analysis"):
            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

            st.title("Top Satistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # Monthly Timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green', marker=".")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily Timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green', marker=".")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Activity map

            st.title("Activity Map")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most Busy Day")
                busy_day = helper.week_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='c')
                plt.xticks(rotation='vertical')
                helper.addlabels(busy_day.index, busy_day.values)  # To add data labels on bar graph
                st.pyplot(fig)

            with col2:
                st.header("Most Busy Month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='#c20078')
                plt.xticks(rotation='vertical')
                helper.addlabels(busy_month.index, busy_month.values)  # To add data labels on bar graph
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            fig, ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # finding the busiest user in the group(Group Level)
            if selected_user == 'Overall':
                st.title("Most Busiest User")
                x, new_df = helper.most_busy_user(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)



            # Most Common Words
            most_common_df = helper.most_common_words(selected_user, df)
            st.title("Most Common Words")
            if most_common_df.empty:
                st.write("<span style='font-size: 24px'>🚫 Words are not present or irrelevant words in chat.</span>", unsafe_allow_html=True)

            else:
                fig = go.Figure(data=go.Bar(
                    x=most_common_df[1],
                    y=most_common_df[0],
                    orientation='h',  # Set bars to be horizontal
                    text=most_common_df[1],  # Use word counts as data labels
                    textposition='inside',
                    insidetextanchor='middle',
                    marker=dict(color='mediumseagreen')
                ))
                fig.update_traces(textfont=dict(size=14, color='white'))
                # Customize axis labels and tick values
                fig.update_layout(xaxis_title='Value', yaxis_title='Category',
                                  xaxis=dict(tickmode='auto', tick0=0, dtick=5))

                st.plotly_chart(fig)

            # WordCloud
            st.title("WordCloud")
            df_wc = helper.create_word_cloud(selected_user, df)
            if df_wc is None:
                st.write("<span style='font-size: 24px'>🚫 Word Cloud is not possible.</span>", unsafe_allow_html=True)
            else:
                fig, ax = plt.subplots()
                ax.imshow(df_wc, interpolation='bilinear')
                st.pyplot(fig)

            # Emoji Analysis
            emoji_df = helper.emoji_helper(selected_user, df)
            st.title("Emoji Analysis")
            if emoji_df.size == 0:
                # Set the font size to 24px
                st.write("<span style='font-size: 24px'>🚫 No emojis used in chat.</span>", unsafe_allow_html=True)
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Count Table")
                    st.dataframe(emoji_df)
                with col2:
                    fig, ax = plt.subplots()

                    # Create a pie chart
                    trace = go.Pie(labels=emoji_df['Emojis'].head(), values=emoji_df['Count'].head(), hole=0.5)
                    # Create a layout for the plot
                    layout = go.Layout()
                    # Create a figure object and add the trace and layout
                    fig = go.Figure(data=[trace], layout=layout)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(width=800, height=500)

                    st.write("Donut Chart of Top 5 Emoji")
                    st.plotly_chart(fig)

            # Sentiment Analysis

            st.title("Sentiment Analysis")
            sentiment = helper.nlp_sentiment_analysis(selected_user, df)

            fig, ax = plt.subplots()
            plt.figure(figsize=(10, 5))
            ax.bar(sentiment.index, sentiment.values, color=['yellow', 'green', 'red'])
            # To add data labels
            for i, v in enumerate(sentiment.values):
                ax.text(i, v / 2, str(v), color='white', fontweight='bold', ha='center',
                        bbox=dict(facecolor='black', alpha=0.5))
            plt.xlabel("Sentiment", fontsize=14)
            plt.ylabel("Number of Messages", fontsize=14)
            plt.title("Sentiment Analysis of Messages as Classification", fontsize=18, fontweight='bold')
            st.pyplot(fig)

            # Early Bird & Night Owl Detection
            st.title("Fun Facts")

            # Identify the person with the most messages sent during night hours
            night_owl = df[(df['hour'] >= 0) & (df['hour'] < 5)].groupby('user').size().idxmax()

            # Identify the person with the most messages sent during early morning hours
            early_bird = df[(df['hour'] >= 5) & (df['hour'] < 9)].groupby('user').size().idxmax()

            # To display message differently in a group and personal chat
            unique_user = df['user'].unique()
            if len(unique_user) > 3:  # For Group Chat
                st.subheader(f"{early_bird} is the early bird in the group.")
                st.subheader(f"{night_owl} is the night owl in the group.")
            else:  # For Personal Chat
                st.subheader(f"{early_bird} is the early bird.")
                st.subheader(f"{night_owl} is the night owl.")

except Exception as e:
    # Display an error message if an exception is raised
    st.error(f"Error occured is: {e}")
