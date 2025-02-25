CREATE TABLE orders (
    Order_ID TEXT,
    Date DATE,
    Warehouse TEXT,
    Dealer TEXT,
    Customer_ID TEXT,
    Machine_ID TEXT,
    Machine_Type TEXT,
    Order_Quantity VARCHAR,
    Production_Status TEXT,
    Inventory_Level INT,
    Change_Type TEXT,
    Order_Volatility TEXT,
    Lead_Time_Days TEXT,
    Delay_Days TEXT
);

drop table orders;

COPY orders FROM 'C:\Users\HP\Desktop\Demand & Production Optimization\Dataset Raw.csv'
DELIMITER ',' 
CSV HEADER;

-- Select Query
select * from orders;

-- Backup Table
CREATE TABLE orders_backup AS TABLE orders;

/* Query to Find Duplicates */
SELECT 
    Order_ID, Date, Warehouse, Dealer, Customer_ID, Machine_ID, Machine_Type, 
    Order_Quantity, Production_Status, Inventory_Level, Change_Type, 
    Order_Volatility, Lead_Time_Days, Delay_Days, 
    COUNT(*) AS duplicate_count
FROM orders
GROUP BY 
    Order_ID, Date, Warehouse, Dealer, Customer_ID, Machine_ID, Machine_Type, 
    Order_Quantity, Production_Status, Inventory_Level, Change_Type, 
    Order_Volatility, Lead_Time_Days, Delay_Days
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC;

/* Datatypes Change */

-- 1. Convert Order_ID to INTEGER (if numeric)

ALTER TABLE orders 
ALTER COLUMN Order_ID TYPE INTEGER USING Order_ID::INTEGER;

-- 2. Convert Order_Quantity to INTEGER (if numeric)

ALTER TABLE orders 
ALTER COLUMN Order_Quantity TYPE INTEGER USING Order_Quantity::INTEGER;

-- 3. Convert Lead_Time_Days to INTEGER (if numeric)

ALTER TABLE orders 
ALTER COLUMN Lead_Time_Days TYPE INTEGER USING Lead_Time_Days::INTEGER;

-- 4. Convert Delay_Days to INTEGER (if numeric)

ALTER TABLE orders 
ALTER COLUMN Delay_Days TYPE INTEGER USING Delay_Days::INTEGER;


/* Count of Not Null & Null*/
SELECT 
    'Order_ID' AS column_name, COUNT(*) AS total_rows, COUNT(Order_ID) AS not_null_count, COUNT(*) - COUNT(Order_ID) AS null_count FROM orders

UNION ALL

SELECT 
    'Date', COUNT(*), COUNT(Date), COUNT(*) - COUNT(Date) FROM orders

UNION ALL

SELECT 
    'Warehouse', COUNT(*), COUNT(Warehouse), COUNT(*) - COUNT(Warehouse) FROM orders

UNION ALL

SELECT 
    'Dealer', COUNT(*), COUNT(Dealer), COUNT(*) - COUNT(Dealer) FROM orders

UNION ALL

SELECT 
    'Customer_ID', COUNT(*), COUNT(Customer_ID), COUNT(*) - COUNT(Customer_ID) FROM orders

UNION ALL

SELECT 
    'Machine_ID', COUNT(*), COUNT(Machine_ID), COUNT(*) - COUNT(Machine_ID) FROM orders

UNION ALL

SELECT 
    'Machine_Type', COUNT(*), COUNT(Machine_Type), COUNT(*) - COUNT(Machine_Type) FROM orders

UNION ALL

SELECT 
    'Order_Quantity', COUNT(*), COUNT(Order_Quantity), COUNT(*) - COUNT(Order_Quantity) FROM orders

UNION ALL

SELECT 
    'Production_Status', COUNT(*), COUNT(Production_Status), COUNT(*) - COUNT(Production_Status) FROM orders

UNION ALL

SELECT 
    'Inventory_Level', COUNT(*), COUNT(Inventory_Level), COUNT(*) - COUNT(Inventory_Level) FROM orders

UNION ALL

SELECT 
    'Change_Type', COUNT(*), COUNT(Change_Type), COUNT(*) - COUNT(Change_Type) FROM orders

UNION ALL

SELECT 
    'Order_Volatility', COUNT(*), COUNT(Order_Volatility), COUNT(*) - COUNT(Order_Volatility) FROM orders

UNION ALL

SELECT 
    'Lead_Time_Days', COUNT(*), COUNT(Lead_Time_Days), COUNT(*) - COUNT(Lead_Time_Days) FROM orders

UNION ALL

SELECT 
    'Delay_Days', COUNT(*), COUNT(Delay_Days), COUNT(*) - COUNT(Delay_Days) FROM orders
ORDER BY column_name; 

/* Removing Order ID Nulls */
DELETE FROM orders
WHERE Order_ID IS NULL;

-- Numeric Columns
/* SQL Query for Summary Statistics */
SELECT 
    'Order_Quantity' AS column_name,
    AVG(Order_Quantity::NUMERIC) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Order_Quantity::NUMERIC) AS median,
    STDDEV(Order_Quantity::NUMERIC) AS std,
    MIN(Order_Quantity::NUMERIC) AS min,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Order_Quantity::NUMERIC) AS percentile_25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Order_Quantity::NUMERIC) AS percentile_50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Order_Quantity::NUMERIC) AS percentile_75,
    MAX(Order_Quantity::NUMERIC) AS max
FROM orders

UNION ALL

SELECT 
    'Inventory_Level' AS column_name,
    AVG(Inventory_Level) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Inventory_Level) AS median,
    STDDEV(Inventory_Level) AS std,
    MIN(Inventory_Level) AS min,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Inventory_Level) AS percentile_25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Inventory_Level) AS percentile_50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Inventory_Level) AS percentile_75,
    MAX(Inventory_Level) AS max
FROM orders

UNION ALL

SELECT 
    'Lead_Time_Days' AS column_name,
    AVG(Lead_Time_Days::NUMERIC) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Lead_Time_Days::NUMERIC) AS median,
    STDDEV(Lead_Time_Days::NUMERIC) AS std,
    MIN(Lead_Time_Days::NUMERIC) AS min,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Lead_Time_Days::NUMERIC) AS percentile_25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Lead_Time_Days::NUMERIC) AS percentile_50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Lead_Time_Days::NUMERIC) AS percentile_75,
    MAX(Lead_Time_Days::NUMERIC) AS max
FROM orders

UNION ALL

SELECT 
    'Delay_Days' AS column_name,
    AVG(Delay_Days) AS mean,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Delay_Days) AS median,
    STDDEV(Delay_Days) AS std,
    MIN(Delay_Days) AS min,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY Delay_Days) AS percentile_25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY Delay_Days) AS percentile_50,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY Delay_Days) AS percentile_75,
    MAX(Delay_Days) AS max
FROM orders;


-- Categorical Columns
-- EDA for Categorical Data
WITH Total_Count AS (
    SELECT COUNT(*) AS total_rows FROM orders
)
SELECT 
    'Warehouse' AS column_name,
    COUNT(*) AS total_count,
    COUNT(Warehouse) AS not_null_count,
    COUNT(*) - COUNT(Warehouse) AS null_count,
    COUNT(DISTINCT Warehouse) AS unique_count,
    (SELECT Warehouse FROM orders GROUP BY Warehouse ORDER BY COUNT(*) DESC LIMIT 1) AS most_frequent_category,
    (SELECT COUNT(*) FROM orders WHERE Warehouse = (SELECT Warehouse FROM orders GROUP BY Warehouse ORDER BY COUNT(*) DESC LIMIT 1)) AS most_frequent_count,
    ((SELECT COUNT(*) FROM orders WHERE Warehouse = (SELECT Warehouse FROM orders GROUP BY Warehouse ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count)) AS percentage_distribution
FROM orders

UNION ALL

SELECT 
    'Dealer',
    COUNT(*),
    COUNT(Dealer),
    COUNT(*) - COUNT(Dealer),
    COUNT(DISTINCT Dealer),
    (SELECT Dealer FROM orders GROUP BY Dealer ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Dealer = (SELECT Dealer FROM orders GROUP BY Dealer ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Dealer = (SELECT Dealer FROM orders GROUP BY Dealer ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Customer_ID',
    COUNT(*),
    COUNT(Customer_ID),
    COUNT(*) - COUNT(Customer_ID),
    COUNT(DISTINCT Customer_ID),
    (SELECT Customer_ID FROM orders GROUP BY Customer_ID ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Customer_ID = (SELECT Customer_ID FROM orders GROUP BY Customer_ID ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Customer_ID = (SELECT Customer_ID FROM orders GROUP BY Customer_ID ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Machine_ID',
    COUNT(*),
    COUNT(Machine_ID),
    COUNT(*) - COUNT(Machine_ID),
    COUNT(DISTINCT Machine_ID),
    (SELECT Machine_ID FROM orders GROUP BY Machine_ID ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Machine_ID = (SELECT Machine_ID FROM orders GROUP BY Machine_ID ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Machine_ID = (SELECT Machine_ID FROM orders GROUP BY Machine_ID ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Machine_Type',
    COUNT(*),
    COUNT(Machine_Type),
    COUNT(*) - COUNT(Machine_Type),
    COUNT(DISTINCT Machine_Type),
    (SELECT Machine_Type FROM orders GROUP BY Machine_Type ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Machine_Type = (SELECT Machine_Type FROM orders GROUP BY Machine_Type ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Machine_Type = (SELECT Machine_Type FROM orders GROUP BY Machine_Type ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Production_Status',
    COUNT(*),
    COUNT(Production_Status),
    COUNT(*) - COUNT(Production_Status),
    COUNT(DISTINCT Production_Status),
    (SELECT Production_Status FROM orders GROUP BY Production_Status ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Production_Status = (SELECT Production_Status FROM orders GROUP BY Production_Status ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Production_Status = (SELECT Production_Status FROM orders GROUP BY Production_Status ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Change_Type',
    COUNT(*),
    COUNT(Change_Type),
    COUNT(*) - COUNT(Change_Type),
    COUNT(DISTINCT Change_Type),
    (SELECT Change_Type FROM orders GROUP BY Change_Type ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Change_Type = (SELECT Change_Type FROM orders GROUP BY Change_Type ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Change_Type = (SELECT Change_Type FROM orders GROUP BY Change_Type ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders

UNION ALL

SELECT 
    'Order_Volatility',
    COUNT(*),
    COUNT(Order_Volatility),
    COUNT(*) - COUNT(Order_Volatility),
    COUNT(DISTINCT Order_Volatility),
    (SELECT Order_Volatility FROM orders GROUP BY Order_Volatility ORDER BY COUNT(*) DESC LIMIT 1),
    (SELECT COUNT(*) FROM orders WHERE Order_Volatility = (SELECT Order_Volatility FROM orders GROUP BY Order_Volatility ORDER BY COUNT(*) DESC LIMIT 1)),
    ((SELECT COUNT(*) FROM orders WHERE Order_Volatility = (SELECT Order_Volatility FROM orders GROUP BY Order_Volatility ORDER BY COUNT(*) DESC LIMIT 1)) * 100.0 / (SELECT total_rows FROM Total_Count))
FROM orders;


