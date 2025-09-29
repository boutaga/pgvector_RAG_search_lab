-- Create source schema and tables for Northwind data
-- This creates the normalized operational schema that will be analyzed by RAG

-- Create the source schema
DROP SCHEMA IF EXISTS src_northwind CASCADE;
CREATE SCHEMA src_northwind;

-- Categories table
CREATE TABLE src_northwind.categories (
    category_id INTEGER PRIMARY KEY,
    category_name VARCHAR(15) NOT NULL,
    description TEXT,
    picture BYTEA
);

-- Suppliers table
CREATE TABLE src_northwind.suppliers (
    supplier_id INTEGER PRIMARY KEY,
    company_name VARCHAR(40) NOT NULL,
    contact_name VARCHAR(30),
    contact_title VARCHAR(30),
    address VARCHAR(60),
    city VARCHAR(30),
    region VARCHAR(30),
    postal_code VARCHAR(20),
    country VARCHAR(30),
    phone VARCHAR(24),
    fax VARCHAR(24),
    homepage TEXT
);

-- Products table
CREATE TABLE src_northwind.products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(40) NOT NULL,
    supplier_id INTEGER REFERENCES src_northwind.suppliers(supplier_id),
    category_id INTEGER REFERENCES src_northwind.categories(category_id),
    quantity_per_unit VARCHAR(20),
    unit_price NUMERIC(10,2),
    units_in_stock INTEGER,
    units_on_order INTEGER,
    reorder_level INTEGER,
    discontinued BOOLEAN DEFAULT false
);

-- Regions table
CREATE TABLE src_northwind.regions (
    region_id INTEGER PRIMARY KEY,
    region_description VARCHAR(50) NOT NULL
);

-- Territories table
CREATE TABLE src_northwind.territories (
    territory_id VARCHAR(20) PRIMARY KEY,
    territory_description VARCHAR(50) NOT NULL,
    region_id INTEGER REFERENCES src_northwind.regions(region_id)
);

-- Employees table
CREATE TABLE src_northwind.employees (
    employee_id INTEGER PRIMARY KEY,
    last_name VARCHAR(20) NOT NULL,
    first_name VARCHAR(10) NOT NULL,
    title VARCHAR(30),
    title_of_courtesy VARCHAR(25),
    birth_date DATE,
    hire_date DATE,
    address VARCHAR(60),
    city VARCHAR(30),
    region VARCHAR(30),
    postal_code VARCHAR(20),
    country VARCHAR(30),
    home_phone VARCHAR(24),
    extension VARCHAR(4),
    photo BYTEA,
    notes TEXT,
    reports_to INTEGER REFERENCES src_northwind.employees(employee_id),
    photo_path VARCHAR(255)
);

-- Employee Territories junction table
CREATE TABLE src_northwind.employee_territories (
    employee_id INTEGER REFERENCES src_northwind.employees(employee_id),
    territory_id VARCHAR(20) REFERENCES src_northwind.territories(territory_id),
    PRIMARY KEY (employee_id, territory_id)
);

-- Customers table
CREATE TABLE src_northwind.customers (
    customer_id VARCHAR(5) PRIMARY KEY,
    company_name VARCHAR(40) NOT NULL,
    contact_name VARCHAR(30),
    contact_title VARCHAR(30),
    address VARCHAR(60),
    city VARCHAR(30),
    region VARCHAR(30),
    postal_code VARCHAR(20),
    country VARCHAR(30),
    phone VARCHAR(24),
    fax VARCHAR(24)
);

-- Shippers table
CREATE TABLE src_northwind.shippers (
    shipper_id INTEGER PRIMARY KEY,
    company_name VARCHAR(40) NOT NULL,
    phone VARCHAR(24)
);

-- Orders table
CREATE TABLE src_northwind.orders (
    order_id INTEGER PRIMARY KEY,
    customer_id VARCHAR(5) REFERENCES src_northwind.customers(customer_id),
    employee_id INTEGER REFERENCES src_northwind.employees(employee_id),
    order_date DATE,
    required_date DATE,
    shipped_date DATE,
    ship_via INTEGER REFERENCES src_northwind.shippers(shipper_id),
    freight NUMERIC(10,2),
    ship_name VARCHAR(40),
    ship_address VARCHAR(60),
    ship_city VARCHAR(30),
    ship_region VARCHAR(30),
    ship_postal_code VARCHAR(20),
    ship_country VARCHAR(30)
);

-- Order Details table (fact table)
CREATE TABLE src_northwind.order_details (
    order_id INTEGER REFERENCES src_northwind.orders(order_id),
    product_id INTEGER REFERENCES src_northwind.products(product_id),
    unit_price NUMERIC(10,2) NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    discount NUMERIC(4,2) DEFAULT 0,
    PRIMARY KEY (order_id, product_id)
);

-- Create indexes for foreign keys to improve join performance
CREATE INDEX idx_products_supplier_id ON src_northwind.products(supplier_id);
CREATE INDEX idx_products_category_id ON src_northwind.products(category_id);
CREATE INDEX idx_territories_region_id ON src_northwind.territories(region_id);
CREATE INDEX idx_employee_territories_employee_id ON src_northwind.employee_territories(employee_id);
CREATE INDEX idx_employee_territories_territory_id ON src_northwind.employee_territories(territory_id);
CREATE INDEX idx_orders_customer_id ON src_northwind.orders(customer_id);
CREATE INDEX idx_orders_employee_id ON src_northwind.orders(employee_id);
CREATE INDEX idx_orders_ship_via ON src_northwind.orders(ship_via);
CREATE INDEX idx_order_details_order_id ON src_northwind.order_details(order_id);
CREATE INDEX idx_order_details_product_id ON src_northwind.order_details(product_id);

-- Add comments to tables for metadata generation
COMMENT ON TABLE src_northwind.categories IS 'Product categories with descriptions';
COMMENT ON TABLE src_northwind.suppliers IS 'Suppliers providing products';
COMMENT ON TABLE src_northwind.products IS 'Products available for sale';
COMMENT ON TABLE src_northwind.customers IS 'Customer information';
COMMENT ON TABLE src_northwind.orders IS 'Customer orders';
COMMENT ON TABLE src_northwind.order_details IS 'Line items for each order - main fact table';
COMMENT ON TABLE src_northwind.employees IS 'Employee information';
COMMENT ON TABLE src_northwind.territories IS 'Sales territories';
COMMENT ON TABLE src_northwind.regions IS 'Geographic regions for territories';
COMMENT ON TABLE src_northwind.shippers IS 'Shipping companies';

-- Add column comments for key business metrics
COMMENT ON COLUMN src_northwind.order_details.unit_price IS 'Price per unit at time of order';
COMMENT ON COLUMN src_northwind.order_details.quantity IS 'Number of units ordered';
COMMENT ON COLUMN src_northwind.order_details.discount IS 'Discount percentage applied';
COMMENT ON COLUMN src_northwind.products.unit_price IS 'Current list price';
COMMENT ON COLUMN src_northwind.products.units_in_stock IS 'Current inventory level';
COMMENT ON COLUMN src_northwind.orders.freight IS 'Shipping cost';
COMMENT ON COLUMN src_northwind.orders.order_date IS 'Date order was placed';
COMMENT ON COLUMN src_northwind.orders.shipped_date IS 'Date order was shipped';