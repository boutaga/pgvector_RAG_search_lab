-- Sample KPI queries for demonstration
-- These queries will be used to showcase the mart performance improvements

-- ============================================
-- SALES KPIS
-- ============================================

-- 1. Fastest-selling products (velocity over last 30 days)
-- This query would be optimized in the mart with pre-aggregated data
WITH product_velocity AS (
    SELECT
        p.product_id,
        p.product_name,
        p.category_id,
        SUM(od.quantity) AS total_quantity,
        COUNT(DISTINCT od.order_id) AS order_count,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS revenue
    FROM src_northwind.order_details od
    JOIN src_northwind.orders o ON od.order_id = o.order_id
    JOIN src_northwind.products p ON od.product_id = p.product_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY p.product_id, p.product_name, p.category_id
)
SELECT
    product_name,
    total_quantity,
    ROUND(total_quantity / 30.0, 2) AS daily_velocity,
    order_count,
    ROUND(revenue::numeric, 2) AS revenue_30d
FROM product_velocity
ORDER BY daily_velocity DESC
LIMIT 10;

-- 2. Top revenue-generating products (last quarter)
SELECT
    p.product_name,
    c.category_name,
    SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue,
    SUM(od.quantity) AS units_sold,
    AVG(od.unit_price) AS avg_price,
    COUNT(DISTINCT od.order_id) AS order_count
FROM src_northwind.order_details od
JOIN src_northwind.orders o ON od.order_id = o.order_id
JOIN src_northwind.products p ON od.product_id = p.product_id
JOIN src_northwind.categories c ON p.category_id = c.category_id
WHERE o.order_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
    AND o.order_date < DATE_TRUNC('quarter', CURRENT_DATE)
GROUP BY p.product_name, c.category_name
ORDER BY total_revenue DESC
LIMIT 20;

-- 3. Customer lifetime value
WITH customer_metrics AS (
    SELECT
        c.customer_id,
        c.company_name,
        MIN(o.order_date) AS first_order_date,
        MAX(o.order_date) AS last_order_date,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS lifetime_value
    FROM src_northwind.customers c
    JOIN src_northwind.orders o ON c.customer_id = o.customer_id
    JOIN src_northwind.order_details od ON o.order_id = od.order_id
    GROUP BY c.customer_id, c.company_name
)
SELECT
    company_name,
    lifetime_value,
    total_orders,
    ROUND(lifetime_value / total_orders, 2) AS avg_order_value,
    (last_order_date - first_order_date) AS customer_tenure_days,
    CASE
        WHEN last_order_date >= CURRENT_DATE - INTERVAL '90 days' THEN 'Active'
        WHEN last_order_date >= CURRENT_DATE - INTERVAL '180 days' THEN 'At Risk'
        ELSE 'Churned'
    END AS customer_status
FROM customer_metrics
ORDER BY lifetime_value DESC
LIMIT 25;

-- ============================================
-- INVENTORY KPIS
-- ============================================

-- 4. Inventory turnover by category
WITH inventory_metrics AS (
    SELECT
        c.category_name,
        SUM(p.units_in_stock * p.unit_price) AS inventory_value,
        SUM(p.units_in_stock) AS total_units,
        COUNT(DISTINCT p.product_id) AS product_count
    FROM src_northwind.products p
    JOIN src_northwind.categories c ON p.category_id = c.category_id
    WHERE p.discontinued = false
    GROUP BY c.category_name
),
sales_metrics AS (
    SELECT
        c.category_name,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS revenue_90d,
        SUM(od.quantity) AS units_sold_90d
    FROM src_northwind.order_details od
    JOIN src_northwind.orders o ON od.order_id = o.order_id
    JOIN src_northwind.products p ON od.product_id = p.product_id
    JOIN src_northwind.categories c ON p.category_id = c.category_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY c.category_name
)
SELECT
    i.category_name,
    i.inventory_value,
    i.total_units,
    s.revenue_90d,
    s.units_sold_90d,
    ROUND((s.revenue_90d * 4) / NULLIF(i.inventory_value, 0), 2) AS annual_turnover_rate,
    ROUND(i.total_units::numeric / NULLIF(s.units_sold_90d::numeric / 90, 0), 1) AS days_of_supply
FROM inventory_metrics i
LEFT JOIN sales_metrics s ON i.category_name = s.category_name
ORDER BY annual_turnover_rate DESC NULLS LAST;

-- 5. Products below reorder level
SELECT
    p.product_name,
    p.units_in_stock,
    p.reorder_level,
    p.units_on_order,
    s.company_name AS supplier,
    c.category_name,
    p.units_in_stock - p.reorder_level AS stock_deficit
FROM src_northwind.products p
JOIN src_northwind.suppliers s ON p.supplier_id = s.supplier_id
JOIN src_northwind.categories c ON p.category_id = c.category_id
WHERE p.units_in_stock < p.reorder_level
    AND p.discontinued = false
ORDER BY stock_deficit;

-- ============================================
-- EMPLOYEE PERFORMANCE KPIS
-- ============================================

-- 6. Top performing sales employees
WITH employee_sales AS (
    SELECT
        e.employee_id,
        e.first_name || ' ' || e.last_name AS employee_name,
        e.title,
        COUNT(DISTINCT o.order_id) AS total_orders,
        COUNT(DISTINCT o.customer_id) AS unique_customers,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue,
        AVG(od.quantity * od.unit_price * (1 - od.discount)) AS avg_order_value
    FROM src_northwind.employees e
    JOIN src_northwind.orders o ON e.employee_id = o.employee_id
    JOIN src_northwind.order_details od ON o.order_id = od.order_id
    WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '3 months')
    GROUP BY e.employee_id, e.first_name, e.last_name, e.title
)
SELECT
    employee_name,
    title,
    total_orders,
    unique_customers,
    ROUND(total_revenue::numeric, 2) AS total_revenue,
    ROUND(avg_order_value::numeric, 2) AS avg_order_value,
    ROUND(total_revenue::numeric / total_orders, 2) AS revenue_per_order
FROM employee_sales
ORDER BY total_revenue DESC;

-- ============================================
-- SHIPPING & LOGISTICS KPIS
-- ============================================

-- 7. Shipping performance by carrier
WITH shipping_metrics AS (
    SELECT
        s.company_name AS shipper,
        COUNT(o.order_id) AS total_shipments,
        AVG(o.shipped_date - o.order_date) AS avg_days_to_ship,
        AVG(o.freight) AS avg_freight_cost,
        SUM(o.freight) AS total_freight_cost,
        STDDEV(o.shipped_date - o.order_date) AS ship_time_variance
    FROM src_northwind.orders o
    JOIN src_northwind.shippers s ON o.ship_via = s.shipper_id
    WHERE o.shipped_date IS NOT NULL
        AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'
    GROUP BY s.company_name
)
SELECT
    shipper,
    total_shipments,
    ROUND(avg_days_to_ship::numeric, 1) AS avg_days_to_ship,
    ROUND(avg_freight_cost::numeric, 2) AS avg_freight_cost,
    ROUND(total_freight_cost::numeric, 2) AS total_freight_cost,
    ROUND(ship_time_variance::numeric, 2) AS ship_time_variance,
    CASE
        WHEN avg_days_to_ship <= 3 THEN 'Excellent'
        WHEN avg_days_to_ship <= 5 THEN 'Good'
        WHEN avg_days_to_ship <= 7 THEN 'Average'
        ELSE 'Poor'
    END AS performance_rating
FROM shipping_metrics
ORDER BY avg_days_to_ship;

-- ============================================
-- TREND ANALYSIS KPIS
-- ============================================

-- 8. Monthly sales trend
SELECT
    DATE_TRUNC('month', o.order_date) AS month,
    COUNT(DISTINCT o.order_id) AS order_count,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    SUM(od.quantity * od.unit_price * (1 - od.discount)) AS revenue,
    SUM(od.quantity) AS units_sold,
    AVG(od.quantity * od.unit_price * (1 - od.discount)) AS avg_order_value
FROM src_northwind.orders o
JOIN src_northwind.order_details od ON o.order_id = od.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '12 months'
GROUP BY DATE_TRUNC('month', o.order_date)
ORDER BY month DESC;

-- 9. Category performance comparison
WITH category_performance AS (
    SELECT
        c.category_name,
        DATE_TRUNC('quarter', o.order_date) AS quarter,
        SUM(od.quantity * od.unit_price * (1 - od.discount)) AS revenue,
        SUM(od.quantity) AS units_sold,
        COUNT(DISTINCT od.order_id) AS order_count
    FROM src_northwind.order_details od
    JOIN src_northwind.orders o ON od.order_id = o.order_id
    JOIN src_northwind.products p ON od.product_id = p.product_id
    JOIN src_northwind.categories c ON p.category_id = c.category_id
    WHERE o.order_date >= DATE_TRUNC('year', CURRENT_DATE)
    GROUP BY c.category_name, DATE_TRUNC('quarter', o.order_date)
)
SELECT
    category_name,
    quarter,
    revenue,
    units_sold,
    order_count,
    LAG(revenue) OVER (PARTITION BY category_name ORDER BY quarter) AS prev_quarter_revenue,
    ROUND(
        ((revenue - LAG(revenue) OVER (PARTITION BY category_name ORDER BY quarter)) /
        NULLIF(LAG(revenue) OVER (PARTITION BY category_name ORDER BY quarter), 0)) * 100,
        2
    ) AS quarter_over_quarter_growth
FROM category_performance
ORDER BY category_name, quarter;

-- 10. Geographic sales distribution
SELECT
    o.ship_country,
    o.ship_region,
    COUNT(DISTINCT o.order_id) AS order_count,
    COUNT(DISTINCT o.customer_id) AS customer_count,
    SUM(od.quantity * od.unit_price * (1 - od.discount)) AS total_revenue,
    AVG(od.quantity * od.unit_price * (1 - od.discount)) AS avg_order_value,
    AVG(o.freight) AS avg_shipping_cost
FROM src_northwind.orders o
JOIN src_northwind.order_details od ON o.order_id = od.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '180 days'
GROUP BY o.ship_country, o.ship_region
ORDER BY total_revenue DESC;