import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
import pickle
import streamlit as st
import os
from datetime import datetime
import squarify
import base64

# GUI setup
st.title("Data Science & Machine Learning Project")
st.header("Customer Segmentation", divider='rainbow')

menu = ["Business Understanding", "Data Understanding","Data preparation","Modeling & Evaluation","Predict"] # , "BigData: Spark"
choice = st.sidebar.selectbox('Menu', menu)

def load_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded successfully!")
        df = pd.read_csv(uploaded_file, encoding='latin-1', sep='\s+', header=None, names=['Customer_id', 'day', 'Quantity', 'Sales'])
        df.to_csv("CDNOW_master_new.txt", index=False)
        df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
        st.session_state['df'] = df
        return df
    else:
        st.write("Please upload a data file to proceed.")
        return None

# Hàm để tạo liên kết tải xuống CSV
def csv_download_link(df, csv_file_name, download_link_text):
    csv_data = df.to_csv(index=True)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{csv_file_name}">{download_link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)    
# Initializing session state variables
if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Main Menu
if choice == 'Business Understanding':
    st.subheader("Business Objective")
    st.write("""
    ###### Customer segmentation is a fundamental task in marketing and customer relationship management. With the advancements in data analytics and machine learning, it is now possible to group customers into distinct segments with a high degree of precision, allowing businesses to tailor their marketing strategies and offerings to each segment's unique needs and preferences.

    ###### Through this customer segmentation, businesses can achieve:
    - **Personalization**: Tailoring marketing strategies to meet the unique needs of each segment.
    - **Optimization**: Efficient allocation of marketing resources.
    - **Insight**: Gaining a deeper understanding of the customer base.
    - **Engagement**: Enhancing customer engagement and satisfaction.

    ###### => Problem/Requirement: Utilize machine learning and data analysis techniques in Python to perform customer segmentation.
    """)
    st.image("Customer-Segmentation.png", caption="Customer Segmentation", use_column_width=True)

    
elif choice == 'Data Understanding':    

    # Liệt kê tất cả các file trong thư mục 'sample_data'
    sample_files = os.listdir('data')
    
    # Tạo một radio button để cho phép người dùng chọn giữa việc sử dụng file mẫu hoặc tải lên file mới
    data_source = st.sidebar.radio('Data source', ['Use a sample file', 'Upload a new file'])
    
    if data_source == 'Use a sample file':
        # Cho phép người dùng chọn một file từ danh sách
        selected_file = st.sidebar.selectbox('Choose a sample file', sample_files)
        
        # Đọc file được chọn (bạn sẽ cần thêm logic để đọc file tại đây)
        file_path = os.path.join('data', selected_file)
        st.session_state['uploaded_file'] = open(file_path, 'r')
        load_data(st.session_state['uploaded_file'])

    else:
        # Cho phép người dùng tải lên một file mới
        st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
        
        if st.session_state['uploaded_file'] is not None:
            load_data(st.session_state['uploaded_file'])

    # st.session_state['uploaded_file'] = st.sidebar.file_uploader("Choose a file", type=['txt'])
    # load_data(st.session_state['uploaded_file'])
    
    if st.session_state['df'] is not None:
        st.write("### Data Overview")
        st.write("Number of rows:", st.session_state['df'].shape[0])
        st.write("Number of columns:", st.session_state['df'].shape[1])
        st.write("First five rows of the data:")
        st.write(st.session_state['df'].head())

elif choice == 'Data preparation': 
    st.write("### Data Cleaning")
    
    if st.session_state['df'] is not None:
        # 1. Handling missing, null, and duplicate values
        st.write("Number of missing values:")
        st.write(st.session_state['df'].isnull().sum())

        st.write("Number of NA values:")
        st.write((st.session_state['df'] == 'NA').sum())

        st.write("Number of duplicate rows:", st.session_state['df'].duplicated().sum())

        # Providing options for handling missing and duplicate values
        if st.checkbox('Remove duplicate rows'):
            st.session_state['df'].drop_duplicates(inplace=True)
            st.write("Duplicate rows removed.")
        
        if st.checkbox('Remove rows with NA values'):
            st.session_state['df'].replace('NA', pd.NA, inplace=True)
            st.session_state['df'].dropna(inplace=True)
            st.write("Rows with NA values removed.")

        # 2. Display number of unique values for each column
        st.write("Number of unique values for each column:")
        st.write(st.session_state['df'].nunique())

        # 3. Plotting distribution for numeric columns
        st.write("### Distribution plots")
        for col in st.session_state['df'].select_dtypes(include=['number']).columns:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            st.session_state['df'][col].hist(ax=ax)
            st.pyplot(fig)

        # 4. Display boxplots for numeric columns
        st.write("### Boxplots for numeric columns")
        for col in st.session_state['df'].select_dtypes(include=['number']).columns:
            st.write(f"#### {col}")
            fig, ax = plt.subplots()
            st.session_state['df'].boxplot(column=col, ax=ax)
            st.pyplot(fig)

        # Additional Data Overview
        st.write("Transactions timeframe from {} to {}".format(st.session_state['df']['day'].min(), st.session_state['df']['day'].max()))
        st.write("{:,} transactions don't have a customer id".format(st.session_state['df'][st.session_state['df'].Customer_id.isnull()].shape[0]))
        st.write("{:,} unique customer_id".format(len(st.session_state['df'].Customer_id.unique())))

        # Add Data Transformation ['Customer_id', 'day', 'Quantity', 'Sales']
        st.write("### Data Transformation")
        # Group the data by Customer_id and sum the other columns, excluding 'day'
        user_grouped = st.session_state['df'].groupby('Customer_id').agg({'Quantity': 'sum', 'Sales': 'sum'})
        st.write("### User Grouped Data")
        st.write(user_grouped.head())

        # Create a new column for the month
        st.session_state['df']['month'] = st.session_state['df']['day'].values.astype('datetime64[M]')
        st.write("### Data with Month Column")
        st.write(st.session_state['df'].head())

        # Plot the total Sales per month
        st.write("### Total Sales per Month")
        dfm = st.session_state['df'].groupby('month')['Quantity'].sum()
        st.line_chart(dfm)

        # Plot the total Quantity per month
        st.write("### Total Quantity per Month")
        dfpc = st.session_state['df'].groupby('month')['Sales'].sum()
        st.line_chart(dfpc)

        # ... (rest of your code, don't forget to modify scatter plots too)

        st.write("### Scatter Plot: Sales vs Quantity for Individual Transactions")
        fig, ax = plt.subplots()
        ax.scatter(st.session_state['df']['Sales'], st.session_state['df']['Quantity'])
        st.pyplot(fig)

        st.write("### Scatter Plot: Sales vs Quantity for User Grouped Data")
        fig, ax = plt.subplots()
        ax.scatter(user_grouped['Sales'], user_grouped['Quantity'])
        st.pyplot(fig)

        # User Feedback section
        st.write("### User Feedback")
        user_feedback = st.text_area("Please share your comments or feedback:", value='')
    
        if st.button("Submit Feedback"):
            # Store the feedback with timestamp in a DataFrame
            current_time = datetime.now()
            feedback_df = pd.DataFrame({
                'Time': [current_time],
                'Feedback': [user_feedback]
            })
    
            # Check if feedback file already exists
            if not os.path.isfile('feedback.csv'):
                feedback_df.to_csv('feedback.csv', index=False)
            else: # Append the new feedback without writing headers
                feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    
            st.success("Your feedback has been recorded!")
    
        # Display the 5 most recent feedbacks
        if os.path.isfile('feedback.csv'):
            all_feedbacks = pd.read_csv('feedback.csv')
            all_feedbacks.sort_values('Time', ascending=False, inplace=True)
            st.write("### 5 Most Recent Feedbacks:")
            st.write(all_feedbacks.head(5))
    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")
    
elif choice == 'Modeling & Evaluation':
    st.write("### Modeling With KMeans")
    if st.session_state['df'] is not None:
        # RFM Analysis
        recent_date = st.session_state['df']['day'].max()

        # Calculate Recency, Frequency, and Monetary value for each customer
        df_RFM = st.session_state['df'].groupby('Customer_id').agg({
            'day': lambda x: (recent_date - x.max()).days, # Recency
            'Customer_id': 'count', # Frequency
            'Sales': 'sum' # Monetary
        }).rename(columns={'day': 'Recency', 'Customer_id': 'Frequency', 'Sales': 'Monetary'})

        st.title('Phân Tích KMeans sử dụng Phương pháp Elbow')

        # Xây dựng và hiển thị biểu đồ Elbow Method
        sse = {}
        for k in range(1, 20):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_RFM)
            sse[k] = kmeans.inertia_

        fig, ax = plt.subplots()
        ax.set_title('Phương pháp Elbow')
        ax.set_xlabel('Số cụm (k)')
        ax.set_ylabel('Tổng Bình phương các khoảng cách')
        sns.pointplot(x=list(sse.keys()), y=list(sse.values()), ax=ax)
        st.pyplot(fig)

        # Cho phép người dùng chọn số lượng cụm k 
        n_clusters = st.sidebar.number_input('Chọn số lượng cụm k từ 2 đến 20:', min_value=2, max_value=20, step=1, key="cluster_value")
        st.write(f'Bạn đã chọn phân thành {n_clusters} cụm.')

        # Áp dụng mô hình KMeans với số lượng cụm đã chọn
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(df_RFM)

        df_sub = df_RFM.copy()
        df_sub['Cluster'] = model.labels_

        # Thống kê mô tả và thống kê theo từng cụm
        cluster_stats = df_sub.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']
        }).round(2)

        cluster_stats.columns = ['RecencyMean', 'FrequencyMean', 'MonetaryMean', 'Count']
        cluster_stats['Percent'] = (cluster_stats['Count'] / cluster_stats['Count'].sum() * 100).round(2)

        # Reset index để 'Cluster' trở thành một cột thông thường, thay vì index
        cluster_stats.reset_index(inplace=True)

        # Đổi tên các nhóm cụm để dễ đọc hơn
        cluster_stats['Cluster'] = 'Cụm ' + cluster_stats['Cluster'].astype('str')

        st.subheader('Thống kê theo từng Cụm')
        st.dataframe(cluster_stats)

        # Biểu đồ Scatter
        fig_scatter = px.scatter(
            cluster_stats,
            x='RecencyMean',
            y='MonetaryMean',
            size='FrequencyMean',
            color='Cluster',
            log_x=True,
            size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Biểu đồ Tree Map
        # Thiết lập màu sắc cho từng cụm - bạn có thể thay đổi này theo ý của bạn
        colors_dict = {
            0: 'green',
            1: 'red',
            2: 'royalblue',
            3: 'orange',
            4: 'purple'
        }
        fig_treemap, ax_treemap = plt.subplots()  # Tạo đối tượng fig và ax riêng biệt cho biểu đồ Tree Map
        fig_treemap.set_size_inches(14, 10)

        squarify.plot(sizes=cluster_stats['Count'], 
                    label=[f'Cụm {i}\n{row.RecencyMean} ngày\n{row.FrequencyMean} đơn hàng\n{row.MonetaryMean} $\n{row.Count} khách hàng ({row.Percent}%)' 
                            for i, row in cluster_stats.iterrows()],
                    color=[colors_dict.get(cluster) for cluster in cluster_stats.index],
                    alpha=0.6,
                    text_kwargs={'fontsize':12, 'fontweight':'bold'})

        ax_treemap.set_title("Phân Khúc Khách Hàng", fontsize=26, fontweight="bold")
        ax_treemap.axis('off')
        st.pyplot(fig_treemap)

        # Vẽ biểu đồ 3D scatter plot
        fig_3d = px.scatter_3d(
            cluster_stats,
            x='RecencyMean',
            y='FrequencyMean',
            z='MonetaryMean',
            color='Cluster',
            size='Count',
            labels={'RecencyMean': 'Recency', 'FrequencyMean': 'Frequency', 'MonetaryMean': 'Monetary'}
        )

        st.plotly_chart(fig_3d, use_container_width=True)

        # Thêm nút để xuất mô hình
        if st.button('Xuất Mô Hình'):
            # Lưu mô hình vào một tập tin .pkl
            with open('kmeans_model.pkl', 'wb') as f:
                pickle.dump((model, cluster_stats), f)
            
            st.session_state.model_exported = True
            st.write('Mô hình (kmeans_model.pkl) đã được xuất thành công!')

        # User Feedback section
        st.write("### User Feedback")
        user_feedback = st.text_area("Please share your comments or feedback:", value='')

        # User Feedback section
        st.write("### User Feedback")
        user_feedback = st.text_area("Please share your comments or feedback:", value='')
    
        if st.button("Submit Feedback"):
            # Store the feedback with timestamp in a DataFrame
            current_time = datetime.now()
            feedback_df = pd.DataFrame({
                'Time': [current_time],
                'Feedback': [user_feedback]
            })
    
            # Check if feedback file already exists
            if not os.path.isfile('feedback.csv'):
                feedback_df.to_csv('feedback.csv', index=False)
            else: # Append the new feedback without writing headers
                feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    
            st.success("Your feedback has been recorded!")
    
        # Display the 5 most recent feedbacks
        if os.path.isfile('feedback.csv'):
            all_feedbacks = pd.read_csv('feedback.csv')
            all_feedbacks.sort_values('Time', ascending=False, inplace=True)
            st.write("### 5 Most Recent Feedbacks:")
            st.write(all_feedbacks.head(5))
    else:
        st.write("No data available. Please upload a file in the 'Data Understanding' section.")

elif choice == 'Predict':
    
    if 'model_exported' in st.session_state and st.session_state.model_exported:
        # Tải lại mô hình và cluster_stats
        with open('kmeans_model.pkl', 'rb') as f:
            model, cluster_stats = pickle.load(f)

        st.subheader('Thống kê theo từng Cụm')
        st.dataframe(cluster_stats)
        
        # Phần mới thêm để nhận dữ liệu từ người dùng và dự đoán
        st.subheader("Dự đoán Cụm cho một Khách hàng mới")
                
        # Nhận dữ liệu từ người dùng
        customer_name = st.text_input('Tên Khách hàng:')
        recent_date = st.date_input('Ngày mua hàng gần nhất:')
        quantity = st.number_input('Số lượng:', min_value=0)
        monetary = st.number_input('Số tiền:', min_value=0.0)
        
        if 'df_new' not in st.session_state:
            st.session_state['df_new'] = pd.DataFrame(columns=['Customer_id', 'day', 'Quantity', 'Sales'])

        if st.button("Add"):
            new_data = pd.DataFrame({'Customer_id': [customer_name], 'day': [recent_date], 'Quantity': [quantity], 'Sales': [monetary]})
            if 'df_new' not in st.session_state:
                st.session_state['df_new'] = new_data
            else:
                st.session_state['df_new'] = pd.concat([st.session_state['df_new'], new_data], ignore_index=True)
            
        st.write("Dữ liệu đã thêm:")
        st.dataframe(st.session_state['df_new'])  # Hiển thị DataFrame sau khi người dùng nhấn "Add"

        # Khi người dùng nhấn nút "Dự đoán", tiến hành dự đoán cụm
        if st.button("Dự đoán"):
            # Tính toán giá trị Recency, Frequency, và Monetary
            recent_date = pd.Timestamp.now().date()  # Cập nhật ngày hiện tại
            df_RFM = st.session_state['df_new'].groupby('Customer_id').agg({
                'day': lambda x: (recent_date - x.max()).days,  # Recency
                'Customer_id': 'count',  # Frequency
                'Sales': 'sum'  # Monetary
            }).rename(columns={'day': 'Recency', 'Customer_id': 'Frequency', 'Sales': 'Monetary'})

            # Dự đoán cụm sử dụng mô hình đã huấn luyện
            cluster_pred = model.predict(df_RFM)
            
            # Thêm cột dự đoán vào df_RFM
            df_RFM['Cluster'] = cluster_pred

            # Hiển thị DataFrame kết quả
            st.write("Kết quả dự đoán:")
            st.dataframe(df_RFM)
            
            # Cho phép người dùng tải xuống kết quả dưới dạng CSV
            csv_download_link(df_RFM, 'RFM_prediction_results.csv', 'Tải xuống kết quả dự đoán')

        # User Feedback section
        st.write("### User Feedback")
        user_feedback = st.text_area("Please share your comments or feedback:", value='')
    
        if st.button("Submit Feedback"):
            # Store the feedback with timestamp in a DataFrame
            current_time = datetime.now()
            feedback_df = pd.DataFrame({
                'Time': [current_time],
                'Feedback': [user_feedback]
            })
    
            # Check if feedback file already exists
            if not os.path.isfile('feedback.csv'):
                feedback_df.to_csv('feedback.csv', index=False)
            else: # Append the new feedback without writing headers
                feedback_df.to_csv('feedback.csv', mode='a', header=False, index=False)
    
            st.success("Your feedback has been recorded!")
    
        # Display the 5 most recent feedbacks
        if os.path.isfile('feedback.csv'):
            all_feedbacks = pd.read_csv('feedback.csv')
            all_feedbacks.sort_values('Time', ascending=False, inplace=True)
            st.write("### 5 Most Recent Feedbacks:")
            st.write(all_feedbacks.head(5))
        
    else:
        st.write("Bạn phải xuất mô hình trước khi tiến hành dự đoán.")
