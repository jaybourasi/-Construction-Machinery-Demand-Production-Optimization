# **Construction Machinery: Demand & Production Optimization**

## **Business Context**
### **Problem Statement**
The client is facing challenges in accurately predicting demand and planning production for various construction machinery variants. Frequent last-minute changes to product features and fluctuating order volumes lead to:
- **Production delays**
- **Inventory imbalance**
- **Difficulties in meeting dealer and customer demand efficiently**

### **Business Objective**
Enhance operational efficiency by improving:
- **Forecasting accuracy**
- **Production planning**
- **Inventory management**

### **Business Constraint**
Minimize **stockouts** and **inventory imbalances** while maintaining stable production.

### **Achievements**
- ✅ Reduced production lead time by 15%
- ✅ Cut inventory holding costs by 10%
- ✅ Optimized demand forecasting and supply chain efficiency
---

## **Approach**
This project followed a structured data-driven methodology using multiple tools:

### 📝 **Excel (Initial Cleaning)**
- Cleaned raw data by **removing duplicates, handling missing values, and formatting** for analysis.

### 🐍 **Python (Data Processing & Analysis)**
- Applied **wrangling techniques** and **feature engineering**.
- Performed **Exploratory Data Analysis (EDA)** for deeper insights.

🔗 **Python Script:** [Click here to access Python code](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Excel%20to%20Python%20(Code).py)

### 🛢 **PostgreSQL (Database Storage & Querying)**
- Stored processed data in **PostgreSQL**.
- Used **SQL queries** to extract valuable insights.

🔗 **SQL Queries:** [Click here to access SQL file](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/EDA%20(SQL%20Code).sql)

### 📊 **Power BI (Data Visualization)**
- Imported database into **Power BI** to create **interactive dashboards** visualizing:
  - Order trends
  - Production delays
  - Inventory utilization

🔗 **Dashboard Images:**
1. [Landing Page](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Landing%20Page.png)
2. [Order Trends & Demand Analysis](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Order%20Trends%20%26%20Demand.png)
3. [Production & Delay Insights](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Production%20%26%20Delay.png)
4. [Inventory Management](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Inventory%20Management.png)

### 📢 **PowerPoint (Presentation)**
- Summarized key insights & recommendations in **PowerPoint slides** for stakeholders.

🔗 **Presentation:** [Click here to view Presentaion PDF](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Jay's%20Final%20Presentation.pdf)

---

## **Dashboards & Key Insights**
### 📌 **1st Dashboard: Order Trends & Demand Analysis**
#### **Key Questions & Insights**
1. **Order Trends**  
   - Highest order volume: **Q4 2023 (524 orders)**  
   - Lowest order volume: **Q1 2023 (478 orders)**
  
2. **Seasonal Demand Trends**  
   - **High Demand:** January, March, October, December  
   - **Low Demand:** February, June  

3. **Warehouse Demand Variation**  
   - **WH002** has the highest demand (**41,993 orders**).  
   - **WH003** has the lowest (**37,111 orders**).

4. **Machine Type Demand**  
   - **Highest orders:** Machine Type 4 (**32,864 orders**)  
   - **Lowest orders:** Machine Type 2 (**31,444 orders**)  

5. **Dealer & Customer Contribution**  
   - **Dealer E & Cust003** contribute the most orders.  

6. **Order Modification Reasons**  
   - **26.1%** changes due to **Delayed Approval**.  
   - **49.6%** changes due to **Urgent Order Changes & Supplier Delay**.  

7. **Production Lead Time & Impact**  
   - **Avg. production lead time:** **29 days**  
   - **On-Time Delivery Rate:** **32.83%**  
   - **Inventory Utilization Rate:** **8.84%** (indicating overstock issues).  

**Order Trends & Demand Analysis:**
![Order Trends & Demand](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Order%20Trends%20%26%20Demand.png)
---

### 📌 **2nd Dashboard: Production & Delay Insights**
#### **Key Questions & Insights**
1. **Production Lead Time**  
   - **Avg. lead time:** **29 days**  

2. **Reasons for Delays**  
   - **36.3%** delays due to **supplier issues**.  
   - **25.2%** orders required **urgent modifications**.  

3. **Delayed Orders Impact**  
   - **67.2%** of total orders faced delays.  
   - **1,952 orders** were delivered on time.  

4. **Production Inefficiencies**  
   - **Machine Type 4** has the highest delayed orders (**838 orders**).  
   - **Machine Type 1** has the least delayed orders (**774 orders**).  

5. **Day-wise Lead Time Trends**  
   - **Highest Lead Time:** **May 7, 2023 (117 days)**  
   - **Lowest Lead Time:** **June 19, 2023 (5 days)**  

**Production & Delay Insights:**
![Production & Delay Insights](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Production%20%26%20Delay.png)
---

### 📌 **3rd Dashboard: Inventory Management**
#### **Key Questions & Insights**
1. **Inventory Utilization Efficiency**  
   - **Utilization rate:** **8.84%** (suggests overstocking).  
   - **Total Inventory:** **2M** vs. **Total Orders:** **158K** (imbalance).  

2. **Stock Trends & Overstocking**  
   - **Overstocking Peaks:** **October 2021, August 2023**  
   - **Understocking Periods:** **February 2022, July 2023**  

3. **Order Modification & Delays by Customers**  
   - **Cust004** has the highest **urgent order changes** (**323 orders**).  
   - **Cust003** has the most **delayed approvals** (**321 orders**).  

4. **Inventory Utilization Trends**  
   - **Highest Utilization Rate:** **August 30, 2023 (42.60%)**  
   - **Lowest Utilization Rate:** **September 2021 (1.29%)**  

5. **Warehouse Performance**  
   - **Highest Inventory Level:** **WH002 (472,810 units)**  
   - **Lowest Inventory Level:** **WH004 (424,958 units)**  

6. **Order Volatility Over Time**  
   - **2023 had the most high-volatility spikes**.  

**Inventory Management:**
![Inventory Management](https://github.com/jaybourasi/Construction-Machinery-Demand-and-Production-Optimization/blob/main/Images/Inventory%20Management.png)
---

## 📬 Let's Connect!  
Questions are always welcome. Feel free to reach out:  

🔗 **LinkedIn:** [Jay Bourasi](https://www.linkedin.com/in/jaybourasi)  
🌍 **Portfolio:** [jaybourasi010.wixsite.com/my-site](https://jaybourasi010.wixsite.com/my-site)  
📧 **Email:** [jaybourasi010@gmail.com](mailto:jaybourasi010@gmail.com)  
🚀 Let's collaborate and optimize data-driven solutions together!
