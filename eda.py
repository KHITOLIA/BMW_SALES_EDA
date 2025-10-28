import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
import kagglehub
import seaborn as sns
# ------------------------------------------------
# Streamlit page setup
# ------------------------------------------------
st.set_page_config(
    page_title="BMW Sales EDA",
    page_icon="🚗",
    layout="wide"
)
le = LabelEncoder()

# Download latest version
path = kagglehub.dataset_download("ahmadrazakashif/bmw-worldwide-sales-records-20102024")

print("Path to dataset files:", path)

for file in os.listdir(path):
  print(file)

df = pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")
df.drop(columns=['Mileage_KM'], inplace=True)

categorical_features = []
numerical_features = []
for col in df.columns:
    if df[col].dtypes == object:
        categorical_features.append(col)
    else:
        numerical_features.append(col)


def dashboard():
    st.sidebar.title("🚗 Exploratory Data Analysis ")
    page = st.sidebar.radio("📂", ["Overview", "Numerical Features Distribution","Categorical Features Distribution", "Pie-Chart","Outlier Detection", "Sales Trend Analysis","Price Trend Analysis", "Regional Analysis"])
    if page == "Overview":
       
       st.title("Dataset Overview")
       st.dataframe(data = df)
       st.subheader("Statistical Summary")
       st.write(df.describe())
       st.write("### Categorical Value Counts")
       st.write(df['Model'].value_counts())
       st.write(df['Region'].value_counts())
       st.write(df['Color'].value_counts())
       st.write(df['Fuel_Type'].value_counts())
       st.write(df['Transmission'].value_counts())
       st.write("### Missing Values in Each Column")
       st.write(df.isnull().sum())

       st.write("### Categorical Features and their Unique Values")
       for col in categorical_features:
        st.write(f"{col}")
        st.write(sorted(df[col].unique()))
        st.write("")
       st.subheader("Insights")
       st.write(f'''1. Data contains 50000 rows and 11 columns
                    \n2. The dataset contains a mix of categorical and numerical features.
                    \n3. Numerical features include {numerical_features[0]}, {numerical_features[1]}, {numerical_features[2]}, {numerical_features[3]}. 
                    \n4. The dataset appears to be well-structured with no missing values in the displayed overview.
                    \n5. Categorical features include Model, Region, Color, Fuel_Type, and Transmission.
                    \n6. According to this data bwm sold their cars from 2010-2024
                    \n7. Their Engine size varies from 1.5L-5.0L
                    \n8. Price range of cars is from 20,000 USD to 150,000 USD
                    ''')
       
    elif page == "Numerical Features Distribution":
        st.title("Distribution of Numerical Features")
        for col in numerical_features:
            plt.figure(figsize=(8, 3))
            sns.histplot(df[col], kde=True, color="blue")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. data has been properly in the format no negative values are present here
                    \n2. All features have a uniform distribution.
                    \n3. from the above analysis we can say that no need of mileage col no relationship found.
                    \n4. hence we can drop the mileage column from the data for better analysis
                    ''')

    elif page == 'Categorical Features Distribution':
        st.title("Categorical Features Analysis")
        for col in categorical_features:
            plt.figure(figsize= (8, 5))
            ax = sns.countplot(x = col, data = df, palette="magma", order=df[col].value_counts().reset_index()[col])        
            for cont in ax.containers:
                ax.bar_label(cont)
            plt.title(f"{col} Distribution ")
            plt.xticks(rotation = 50)
            plt.xlabel(f'{col}')
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        
        st.subheader("Insights")
        st.write('''1. all categories in each categorical col are equally distributed
                    \n2. Model: The dataset includes a variety of BMW models, with some models being more popular than others.
                    \n3. Region: Sales are distributed across multiple regions, indicating a global presence.''')
    
    elif page == "Pie-Chart":
        st.title("Categorical Features Pie-Chart Analysis")
        for col in categorical_features:
            plt.figure(figsize=(8, 5))
            pie_data = df[col].value_counts()
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
            plt.title(f"{col} Distribution ")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        
        st.subheader("Insights")
        st.write('''1. All categories in each categorical column are fairly represented in the dataset.
                    \n2. The distribution of car models, regions, colors, fuel types, and transmissions appears balanced without extreme dominance by any single category.
                    \n3. This balance is beneficial for building robust machine learning models as it reduces bias towards any particular category.
                    ''')
    
    elif page == "Outlier Detection":
        st.title("Outlier Detection using Box Plots")
        for col in numerical_features:
            plt.figure(figsize=(8, 3))
            sns.boxplot(y = col, data = df, color="orange")
            plt.title(f"{col} Distribution ")
            plt.xlabel(f'{col}')
            plt.ylabel("Values")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. No outliers are present in the numerical features as per the box plot analysis
                    \n2. All the columns are in proper type
                    \n3. No need of outlier treatment as no outliers are present in the data
                    \n4. Data is clean and ready for further analysis or modeling
                    \n5. Engine size is mostly concentrated around 2000-3500 CC''')
        
    elif page == "Sales Trend Analysis":
        
        st.title("Sales Trend by features")
        st.write("Analyzing sales trends based on different features.")
        tabs = st.tabs(['Categorical Features', "Numerical Features"])
        with tabs[0]:
            st.subheader("Sales Trend by Categorical Features")
            for col in categorical_features:
                idx = categorical_features.index(col)
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x = col, y = 'Sales_Volume', data = df,ci = None,  estimator = sum, order = df.groupby(col)['Sales_Volume'].sum().sort_values(ascending=False).index, palette='magma')
                for cont in ax.containers:
                    ax.bar_label(cont, fmt='%.0f')
                idx = idx + 1
                plt.title(f"Total BMW Sales by {col}")
                plt.tight_layout()
                plt.show()
                st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. BMW is selling almost same amount of different Models very minute difference in the sales volume per model.
                    \n2. It clearly reflecting the behaviour of customers are showing interest not only to specific model but also to every model equally.
                    \n3. Mostly Asians are Beemer Lover
                    \n4. BMW Sales is approximately same among rest of the regions
                    \n6. Customers are more interested in Hybrid type of cars because they might gives more mileage
                    \n7.  Manual Transmission Cars leading the race''')
     
        with tabs[1]:
            st.subheader("Sales Trend by Numerical Features")
            for col in numerical_features:
                if col == 'Sales_Volume':
                    pass
                elif col == 'Price_USD':
                    pass
                else:

                    plt.figure(figsize=(10, 6))
                    ax = sns.lineplot(x = col, y = 'Sales_Volume', data = df, ci = None, estimator= sum, palette='magma')
                    for cont in ax.containers:
                        ax.bar_label(cont, fmt='%.0f')  
                    plt.title(f"Total BMW Sales by {col}")
                    plt.tight_layout()
                    plt.show()
                    st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. BMW sold almost same amount of cars from 2010 to 2017 but after 2018 it leads to showing some alternate fluctuations per year till 2024 by increasing lately
                        \n2. Peak Sales goes in 2022 but slightly decrease in next two years.
                        \n3 . The above plot is uniformly distributed, varing from 1.5L-5L car Engine means almost all type of engine size had been sold equally.''')
        
        # with tabs[2]:
        #     sns.lineplot(x = 'Year', y = 'Sales_Volume', hue='Region', data = df, ci = None)            st.pyplot(plt)
        #     st.pyplot(plt)

    elif page == "Price Trend Analysis":
        
        st.title("Price Trend by features")
        st.write("Analyzing Price trends based on different features.")
        tabs = st.tabs(['Categorical Features', "Numerical Features"])
        with tabs[0]:
            st.subheader("Price Trend by Categorical Features")
            for col in categorical_features:
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x = col, y = 'Price_USD', data = df,ci = None,  estimator= sum, order = df.groupby(col)['Price_USD'].sum().sort_values(ascending=False).index, palette='magma')
                for cont in ax.containers:
                    ax.bar_label(cont, fmt='%.0f')
                plt.title(f"Total BMW Price by {col}")
                plt.tight_layout()
                plt.show()
                st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. Model: Certain BMW models consistently outperform others in price, indicating strong consumer preference for these models.
                        \n2. Region: Price levels vary significantly by region, suggesting that market strategies should be tailored to regional preferences and economic conditions.
                        \n3. Color: Some car colors are more popular than others, which could influence inventory and marketing decisions.
                        \n4. Fuel_Type: The preference for fuel types (e.g., Petrol, Diesel, Electric) varies, reflecting changing consumer attitudes towards sustainability and fuel efficiency.
                        \n5. Transmission: The choice between Automatic and Manual transmissions shows distinct price patterns, which could impact product offerings.''')

        with tabs[1]:
            st.subheader("Price Trend by Numerical Features")
            for col in numerical_features:
                if col == 'Sales_Volume':
                    pass
                elif col == 'Price_USD':
                    pass
                else:
                    plt.figure(figsize=(10, 6))
                    ax = sns.lineplot(x = col, y = 'Price_USD', data = df, ci = None, estimator= sum, palette='magma')
                    for cont in ax.containers:
                        ax.bar_label(cont, fmt='%.0f')  
                    plt.title(f"Total BMW Price by {col}")
                    plt.tight_layout()
                    plt.show()
                    st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. Year: Price levels show a clear trend over the years,   indicating growth or decline in BMW's market presence.
                        \n2. Engine_Size_CC: There is a correlation between engine size and price, suggesting consumer preferences for certain engine capacities.
                        \n3. Price_USD: Price levels vary across different price points, highlighting the importance of pricing strategies in attracting customers.''')
            
            












dashboard()
