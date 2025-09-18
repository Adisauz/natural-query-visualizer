import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase


app = Flask(__name__)
# Enable CORS for development across LAN (frontend on localhost:3000 or any 192.168.x.x:3000)
# You can tighten this later by setting explicit origins via environment if needed
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable is not set!")
    print("Please create a .env file with your API key or set the environment variable.")
    exit(1)

# Set the environment variable for LangChain
os.environ["OPENAI_API_KEY"] = api_key

# Database connections
mysql_uri_chinook = "mysql+pymysql://root:sonu1234@localhost:3306/chinook"
mysql_uri_world = "mysql+pymysql://root:sonu1234@localhost:3306/world"
mysql_uri_imdb = "mysql+pymysql://root:sonu1234@localhost:3306/imdb"

# Initialize databases
databases = {
    "chinook": SQLDatabase.from_uri(mysql_uri_chinook),
    "world": SQLDatabase.from_uri(mysql_uri_world),
    "imdb": SQLDatabase.from_uri(mysql_uri_imdb)
}

# Default database
db = databases["chinook"]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Narrative generation prompt for charts (single or multiple)
narrative_prompt = ChatPromptTemplate.from_template("""
You are a data storytelling expert. Create a flowing narrative for data visualizations.

Question: {question}
Charts: {chart_info}

Create a narrative structure with:

1. **Introduction**: Brief overview of what we're exploring
2. **Chart Transitions**: Connecting text between charts (if multiple charts):
   - What the previous chart showed
   - Why the next chart provides additional insight
   - How they relate to each other
   - For single charts, leave transitions empty: []
3. **Key Insights**: Highlight the most important findings (2-3 insights)
4. **Conclusion**: Summary of the overall story the data tells

Return JSON format:
{{
  "introduction": "Opening paragraph that sets the context",
  "transitions": [
    "Text that connects chart 1 to chart 2",
    "Text that connects chart 2 to chart 3"
  ],
  "insights": [
    "Key insight 1",
    "Key insight 2", 
    "Key insight 3"
  ],
  "conclusion": "Closing paragraph that ties everything together"
}}

**Important**: 
- For single charts: Set "transitions": [] (empty array)
- For multiple charts: Include transition text between each chart
- Keep each text block concise (2-3 sentences) and focus on creating a smooth reading flow
""")

# Chart suggestion prompt for both single and multiple chart generation
chart_suggestion_prompt = ChatPromptTemplate.from_template("""
Analyze this question and suggest 1-4 different chart types that would best visualize the data. Choose the optimal number based on the complexity and nature of the data.

🎯 **CRITICAL: AVOID DATA REDUNDANCY - ENSURE DIVERSE INSIGHTS**
Each chart must provide UNIQUE data perspectives, NOT just different metrics of the same entities:

❌ **FORBIDDEN REDUNDANT PATTERNS:**
- "Top 20 Countries by Population" + "Top 20 Countries by GNP" (same countries, different metrics)
- "Sales by Product A-M" + "Sales by Product N-Z" (same data type, different ranges)
- "Customer Revenue Q1" + "Customer Revenue Q2" (same customers, different periods)

✅ **REQUIRED DIVERSIFICATION STRATEGIES:**
1. **Entity Diversity**: Different countries/customers/products in each chart
2. **Grouping Diversity**: Individual items vs grouped categories vs aggregated summaries
3. **Perspective Diversity**: Geographic vs demographic vs temporal vs performance views
4. **Scale Diversity**: Detailed breakdowns vs high-level overviews

**GOOD DIVERSE EXAMPLES:**
- "Top 20 Countries by Population" + "Population by Continent" + "Population Growth Trends"
- "Sales by Product Category" + "Sales by Region" + "Monthly Sales Trends"
- "Customer Demographics" + "Purchase Patterns" + "Geographic Distribution"

**CHART COMBINATION RULES:**
- If Chart 1 shows individual entities (countries), Chart 2 should show grouped categories (continents)
- If Chart 1 shows current data, Chart 2 should show trends or comparisons  
- If Chart 1 shows one dimension, Chart 2 should show a different dimension entirely

**SPECIFIC EXAMPLES FOR COMMON QUERIES:**
- "European countries analysis" →
  ❌ BAD: "Population of European Countries" + "GNP of European Countries" (redundant)
  ✅ GOOD: "Population of European Countries" + "Population by European Regions" + "Country Size vs Population Density"
  
- "Sales analysis" →
  ❌ BAD: "Sales by Product" + "Revenue by Product" (redundant)
  ✅ GOOD: "Sales by Product Category" + "Sales by Geographic Region" + "Sales Trends Over Time"

Available chart types: bar, line, pie, scatter, table

Guidelines:
- bar → categorical comparisons, counts, rankings, top/bottom lists (best for 3-20 categories)
- line → trends over time, sequential data, year-over-year analysis  
- pie → ONLY for part-to-whole relationships with 3-6 categories MAX (never use for customer lists, individual records, or >6 items)
- scatter → ONLY for correlation/relationship between TWO NUMERIC variables (never use categorical data like names/government forms)
- table → detailed data, individual records, or >10 categories

⚠️ **CRITICAL PIE CHART RESTRICTIONS:**
🚫 **ABSOLUTELY FORBIDDEN PIE CHARTS:**
- Individual customers/people (like "Customer Purchase Distribution")
- Individual products, cities, or detailed records
- ANY data with >6 distinct categories
- Lists of names, IDs, or specific entities

✅ **ONLY ALLOWED PIE CHARTS:**
- High-level categories: Continents (7 max), Product Lines (5-6), Departments
- Geographic regions: North/South/East/West, Major regions only
- Time periods: Quarters (4), Seasons (4), Years (if ≤6)
- Status categories: Active/Inactive, High/Medium/Low

**CRITICAL RULE: If you're tempted to use a pie chart, ask:**
- "Are there more than 6 slices?" → Use bar chart instead
- "Are these individual people/customers?" → Use bar chart instead  
- "Are these specific items/products?" → Use bar chart instead

**GOOD pie chart examples:**
- "Sales by Continent" (7 continents max)
- "Revenue by Product Category" (4-5 categories)
- "Orders by Quarter" (4 quarters)

**BAD pie chart examples (USE BAR INSTEAD):**
- "Customer Purchase Distribution" (59 customers = TERRIBLE!)
- "Sales by Individual Product" (100+ products)
- "Revenue by City" (50+ cities)
- "Purchases by Employee" (20+ employees)

⚠️ **SCATTER CHART REQUIREMENTS:**
🚫 **ABSOLUTELY FORBIDDEN SCATTER CHARTS:**
- Any chart where X or Y axis would be categorical text (names, government forms, categories)
- Charts with only one numeric variable
- Lists of countries/cities without numeric relationships

✅ **ONLY ALLOWED SCATTER CHARTS:**
- Two numeric variables showing correlation: Population vs GNP, Sales vs Profit, Age vs Income
- Numeric performance metrics: Revenue vs Customers, Rating vs Box Office
- Scientific/statistical relationships: Height vs Weight, Temperature vs Sales

**GOOD scatter chart examples:**
- "Population vs Economic Output" (Population on X, GNP on Y)
- "Movie Rating vs Box Office Revenue" (Rating on X, Revenue on Y)
- "Customer Age vs Purchase Amount" (Age on X, Amount on Y)

**BAD scatter chart examples (USE BAR/TABLE INSTEAD):**
- "Countries by Government Form" (Government Form is categorical!)
- "Customer Names vs Purchase History" (Names are not numeric!)
- "Product Categories vs Sales" (Categories are not continuous!)

**CRITICAL RULE: If either axis would show text labels instead of numeric scales, DO NOT use scatter plot!**

⚠️ **TABLE CHART REQUIREMENTS:**
- For table charts, ALWAYS select MULTIPLE DIVERSE columns to show comprehensive data
- NEVER repeat the same column twice in a table
- Include identifying columns (Name, Title, etc.) + multiple data columns (Population, GNP, LifeExpectancy, etc.)
- Example: For countries, select Name, Population, GNP, LifeExpectancy, SurfaceArea (NOT just Name, LifeExpectancy, LifeExpectancy)
- For population data: Show UNIQUE entities (countries/cities), not repetitive regions
- AVOID: Multiple rows with same region/category name - this creates confusing tables
- PREFER: Country-level data over historical/regional breakdowns for tables
- Example: Instead of "Australia and New Zealand" appearing 3 times, show 3 different countries

🔥 **CRITICAL SQL GENERATION RULES:**
- When generating SQL for any chart type, ensure each column in SELECT is UNIQUE
- For detailed/table data: Select 4-6 DIFFERENT columns that provide varied insights
- FORBIDDEN: SELECT a.Name, a.Population, a.Population (duplicate Population)
- REQUIRED: SELECT a.Name, a.Population, a.GNP, a.LifeExpectancy (all different)
- Each column must add new information, not repeat existing data

📊 **CHART-SPECIFIC SQL REQUIREMENTS:**
- **Bar Charts**: For "largest X by Y", ensure ONE result per category (use window functions/subqueries)
- **Pie Charts**: ONLY for high-level groupings with ≤6 categories. Group individual items into broader categories.
  - ✅ GOOD: "SELECT continent, SUM(population) FROM country GROUP BY continent" (7 continents)
  - ❌ BAD: "SELECT customer_name, total_purchases FROM customers" (59 individual customers)
  - **RULE**: If query returns >6 rows, use bar chart instead of pie
- **Tables**: Show UNIQUE entities with comprehensive data - avoid repetitive rows.
  - ✅ GOOD: "SELECT DISTINCT a.Name, a.Population, a.GNP, a.LifeExpectancy FROM country a ORDER BY a.Population DESC LIMIT 50"
  - ✅ GOOD: "SELECT a.continent, AVG(a.population), COUNT(*) as countries FROM country a GROUP BY a.continent"  
  - ❌ BAD: "SELECT a.region, a.year, a.population FROM historicaldata a" (repetitive regions)
  - ❌ BAD: Historical/time-series data that shows same entity multiple times
  - **RULE**: Each row should represent a UNIQUE entity (country, customer, product), not repeated categories

🎯 **CRITICAL: ALWAYS SORT AND LIMIT LARGE DATASETS**
- **ALWAYS use ORDER BY** to get the most important/relevant results first
- **For rankings**: ORDER BY [metric] DESC (highest first)
- **For alphabetical**: ORDER BY [name] ASC  
- **For time series**: ORDER BY [date] DESC (most recent first)
- **Examples**:
  - ✅ "SELECT a.Name, a.Population FROM country a ORDER BY a.Population DESC" (largest countries first)
  - ✅ "SELECT a.title, b.avg_rating FROM movie a JOIN ratings b ON a.id = b.movie_id ORDER BY b.avg_rating DESC" (best movies first)
  - ✅ "SELECT a.Name, SUM(b.Total) FROM customer a JOIN invoice b ON a.CustomerId = b.CustomerId GROUP BY a.Name ORDER BY SUM(b.Total) DESC" (top customers first)
- **Backend will automatically limit results**: Pie (6), Bar (20), Line (50), Scatter (100), Table (50)

Database Schema: {schema}
Question: {question}

Return JSON format with 1-4 suggestions based on data complexity:
{{
  "suggestions": [
    {{
      "chart_type": "bar|line|pie|scatter|table",
      "title": "Descriptive title for this chart",
      "reason": "Why this chart type is useful for this data",
      "sql_focus": "What aspect of the data this chart should focus on"
    }}
    // Add more suggestions only if they provide meaningful additional insights
    // For simple data: 1-2 charts may be sufficient
    // For complex data: 3-4 charts may be needed
  ]
}}
""")

def get_schema(database_name="chinook"):
    """Get database schema information"""
    return databases[database_name].get_table_info()


def get_available_databases():
    """Get list of available databases with their descriptions"""
    return {
        "chinook": "Music store database with artists, albums, tracks, customers, and sales data",
        "world": "Global geographic database with countries, cities, languages, and population data",
        "imdb": "Movie database with films, actors, directors, ratings, and entertainment industry data"
    }

def run_query(query, database_name="chinook"):
    """Execute SQL query on specified database"""
    try:
        return databases[database_name].run(query)
    except Exception as e:
        # Re-raise the exception to be handled by the calling function
        raise e

def run_query_with_columns(query, database_name="chinook"):
    """Execute SQL query and return both data and column names"""
    try:
        # Get the database connection
        db = databases[database_name]
        
        # Execute query to get column names
        import sqlalchemy
        with db._engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            columns = list(result.keys())
            rows = result.fetchall()
            
            # Convert rows to list of tuples
            data = [tuple(row) for row in rows]
            
            return data, columns
    except Exception as e:
        # Fallback to regular run_query
        data = databases[database_name].run(query)
        # Try to infer column names from query
        import re
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', query.upper())
        if select_match:
            select_part = select_match.group(1)
            # Simple column extraction (this is a basic approach)
            columns = [col.strip().split(' AS ')[-1].split('.')[-1] for col in select_part.split(',')]
        else:
            columns = [f'col_{i}' for i in range(len(data[0]) if data else 0)]
        
        return data, columns

# SQL prompt template
sql_prompt = ChatPromptTemplate.from_template(
    """
You are an expert SQL assistant.
Your task is to generate a valid SQL query for a MySQL database. 
The query should run directly in MySQL without any extra formatting (no ```sql ...``` blocks, no explanations).

Follow these strict rules when generating SQL:

⚠️  CRITICAL: ALWAYS use table aliases for ALL column references to avoid ambiguous column errors!

1. **Schema Usage and Validation**
   - CRITICAL: Use ONLY tables and columns that exist in the provided schema
   - Before writing any query, carefully examine the schema to identify:
     * Available table names and their exact spelling
     * Column names and their exact case-sensitive spelling
     * Data types of each column
     * Primary and foreign key relationships
   - Never assume column names based on common database patterns
   - If a requested data point doesn't exist in the schema, use the closest available alternative
   - Always verify that every column referenced in your query exists in the schema

2. **Detailed Table Relationships and Join Patterns**
   
   **For Chinook Music Database:**
   - Core music hierarchy: Artist → Album → Track → InvoiceLine
     * artist.ArtistId = album.ArtistId (one-to-many)
     * album.AlbumId = track.AlbumId (one-to-many)
     * track.TrackId = invoiceline.TrackId (one-to-many)
   - Sales and customer data:
     * customer.CustomerId = invoice.CustomerId (one-to-many)
     * invoice.InvoiceId = invoiceline.InvoiceId (one-to-many)
     * employee.EmployeeId = customer.SupportRepId (one-to-many)
   - Genre classification: genre.GenreId = track.GenreId
   - Media format: mediatype.MediaTypeId = track.MediaTypeId

   **For Store Collectibles Database:**
   - Product catalog hierarchy: productlines → products → orderdetails
     * productlines.productLine = products.productLine (one-to-many)
     * products.productCode = orderdetails.productCode (one-to-many)
   - Order processing flow: customers → orders → orderdetails
     * customers.customerNumber = orders.customerNumber (one-to-many)
     * orders.orderNumber = orderdetails.orderNumber (one-to-many)
   - Payment tracking: customers.customerNumber = payments.customerNumber (one-to-many)
   - Sales organization: employees.employeeNumber = customers.salesRepEmployeeNumber (one-to-many)
     * offices.officeCode = employees.officeCode (one-to-many)

   **For World Geographic Database:**
   - Geographic hierarchy: country → city
     * country.Code = city.CountryCode (one-to-many)
   - Language distribution: country → countrylanguage
     * country.Code = countrylanguage.CountryCode (one-to-many)
   - Capital cities: country.Capital = city.ID (one-to-one)
   
   ⚠️ **CRITICAL WORLD DATABASE RULES:**
   - IndepYear is ONLY in country table (country.IndepYear), NOT in city or countrylanguage
   - Population exists in BOTH country.Population AND city.Population - specify which one!
   - For country population trends over time, use country.IndepYear as time reference
   - countrylanguage table has: CountryCode, Language, IsOfficial, Percentage (NO IndepYear!)
   - city table has: ID, Name, CountryCode, District, Population (NO IndepYear!)
   - NEVER use city.IndepYear or countrylanguage.IndepYear - they don't exist!

   **For IMDB Movie Database:**
   - Movie core data: movie (main table with title, year, duration, etc.)
   - Movie ratings: movie → ratings
     * movie.id = ratings.movie_id (one-to-one)
   - Movie genres: movie → genre
     * movie.id = genre.movie_id (one-to-many)
   - Director relationships: movie → director_mapping → names
     * movie.id = director_mapping.movie_id
     * director_mapping.name_id = names.id
   - Actor/Cast relationships: movie → role_mapping → names
     * movie.id = role_mapping.movie_id
     * role_mapping.name_id = names.id
     * role_mapping.category ('actor', 'actress')

3. **Database-Specific Column Location and Usage Patterns**
   
   **Chinook Database Key Columns:**
   - Sales metrics: invoiceline.Quantity, invoiceline.UnitPrice, invoice.Total
   - Music metadata: track.Name, track.Milliseconds, track.Bytes, album.Title, artist.Name
   - Customer data: customer.FirstName, customer.LastName, customer.Country, customer.City
   - Employee info: employee.FirstName, employee.LastName, employee.Title
   - Time data: invoice.InvoiceDate, track.Milliseconds
   
   **CRITICAL COLUMN LOCATIONS:**
   - Total is ONLY in invoice table (invoice.Total), NOT in invoiceline table
   - Quantity is ONLY in invoiceline table (invoiceline.Quantity), NOT in track table
   - UnitPrice is ONLY in invoiceline table (invoiceline.UnitPrice), NOT in track table
   - Track table has: TrackId, Name, AlbumId, MediaTypeId, GenreId, Composer, Milliseconds, Bytes, UnitPrice
   - InvoiceLine table has: InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity
   - Invoice table has: InvoiceId, CustomerId, InvoiceDate, BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode, Total
   
   ⚠️ **CRITICAL CHINOOK DATABASE RULES:**
   - For customer total purchases, use invoice.Total (NOT invoiceline.Total - it doesn't exist!)
   - Join: customer → invoice → invoiceline (if needed)
   - For revenue calculations: SUM(invoice.Total) for total customer spending
   
   **Store Database Key Columns:**
   - Financial calculations: 
     * Revenue = orderdetails.priceEach * orderdetails.quantityOrdered
     * Total order value = SUM(orderdetails.priceEach * orderdetails.quantityOrdered)
     * Payment amounts = payments.amount
   - Product information:
     * Product identity: products.productCode, products.productName
     * Categorization: products.productLine, productlines.productLine
     * Inventory: products.quantityInStock, products.buyPrice, products.MSRP
     * Specifications: products.productScale, products.productVendor
   - Order tracking:
     * Order timing: orders.orderDate, orders.requiredDate, orders.shippedDate
     * Order status: orders.status
     * Line items: orderdetails.quantityOrdered, orderdetails.orderLineNumber
   - Customer analysis:
     * Customer identity: customers.customerName, customers.contactFirstName, customers.contactLastName
     * Geographic data: customers.city, customers.state, customers.country
     * Credit info: customers.creditLimit
   - Employee and office data:
     * Employee hierarchy: employees.reportsTo (self-referencing)
     * Contact info: employees.email, employees.extension
     * Office locations: offices.city, offices.country, offices.territory
   
   **World Database Key Columns:**
   - Country data:
     * Identity: country.Code, country.Name, country.Code2
     * Geographic: country.Continent, country.Region, country.SurfaceArea
     * Demographics: country.Population, country.LifeExpectancy
     * Economic: country.GNP, country.GNPOld
     * Political: country.GovernmentForm, country.HeadOfState, country.IndepYear
   
   🌍 **CRITICAL WORLD DATABASE REGION FILTERING:**
   - For "European countries", use: WHERE a.Continent = 'Europe' (NOT Region = 'Europe')
   - European regions include: 'Eastern Europe', 'Western Europe', 'Southern Europe', 'Nordic Countries', 'British Islands', 'Baltic Countries'
   - For "Asian countries", use: WHERE a.Continent = 'Asia'
   - For "African countries", use: WHERE a.Continent = 'Africa'
   - For "North American countries", use: WHERE a.Continent = 'North America'
   - For "South American countries", use: WHERE a.Continent = 'South America'
   - ALWAYS use Continent for broad geographic filtering, NOT Region
   - Region is for more specific sub-regions within continents
   - City data:
     * Identity: city.ID, city.Name, city.CountryCode
     * Administrative: city.District
     * Population: city.Population
   - Language data:
     * Language info: countrylanguage.Language, countrylanguage.CountryCode
     * Official status: countrylanguage.IsOfficial ('T' or 'F')
     * Usage: countrylanguage.Percentage
   
   🚫 **FORBIDDEN WORLD DATABASE QUERIES:**
   - NEVER: SELECT b.IndepYear FROM city b (IndepYear doesn't exist in city table!)
   - NEVER: SELECT c.IndepYear FROM countrylanguage c (IndepYear doesn't exist in countrylanguage!)
   - ALWAYS: SELECT a.IndepYear FROM country a (IndepYear only exists in country table)
   - NEVER: WHERE a.Region = 'Europe' (Europe is a Continent, not a Region!)
   - NEVER: WHERE a.Region = 'Asia' (Asia is a Continent, not a Region!)
   - ALWAYS: WHERE a.Continent = 'Europe' (for European countries)
   - ALWAYS: WHERE a.Continent = 'Asia' (for Asian countries)
   
   **IMDB Database Key Columns:**
   - Movie data:
     * Identity: movie.id, movie.title, movie.year
     * Production: movie.date_published, movie.duration, movie.country
     * Financial: movie.worlwide_gross_income
     * Details: movie.languages, movie.production_company
   - Ratings and popularity:
     * Quality metrics: ratings.avg_rating, ratings.median_rating
     * Popularity: ratings.total_votes
   - People data:
     * Identity: names.id, names.name
     * Physical: names.height, names.date_of_birth
     * Career: names.known_for_movies
   - Genre classification:
     * Categories: genre.genre, genre.movie_id
   - Role assignments:
     * Acting roles: role_mapping.category ('actor', 'actress')
     * Connections: role_mapping.movie_id, role_mapping.name_id
   - Director assignments:
     * Director connections: director_mapping.movie_id, director_mapping.name_id
   
   🚫 **FORBIDDEN IMDB DATABASE QUERIES:**
   - NEVER reference non-existent table aliases (like b.title when no table b exists)
   - ALWAYS use correct table aliases in proper alphabetical order (a, b, c, d, e, f)
   - For movie + ratings queries: movie AS a, ratings AS b
   - EXAMPLE CORRECT: SELECT a.title, b.avg_rating FROM movie AS a JOIN ratings AS b ON a.id = b.movie_id
   - EXAMPLE WRONG: SELECT a.year, b.title, c.avg_rating FROM movie AS a JOIN ratings AS c ON a.id = c.movie_id (b doesn't exist!)

4. **Advanced Query Construction Guidelines**
   - For sales analysis in Store DB: Always join through orders → orderdetails → products
   - For revenue calculations: Use orderdetails table for pricing, not products table
   - For product categories: Join products with productlines for descriptive category names
   - For customer analysis: Consider both order history and payment history
   - For inventory analysis: Use products.quantityInStock for current stock levels
   - For employee performance: Count customers assigned via salesRepEmployeeNumber
   - For geographic analysis: Use customer address fields (city, state, country)
   - For time-based analysis: Use orders.orderDate for transaction timing
   
5. **Common Query Patterns by Database**
   
   **Chinook Patterns:**
   - Top artists by sales: artist → album → track → invoiceline (SUM invoiceline.quantity)
   - Top albums by sales: album → track → invoiceline (SUM invoiceline.quantity)
   - Genre popularity: genre → track → invoiceline (SUM invoiceline.quantity)
   - Customer purchasing: customer → invoice → invoiceline (SUM invoiceline.quantity * invoiceline.unitprice)
   - Employee performance: employee → customer → invoice (COUNT/SUM)
   
   **CRITICAL SALES QUERIES:**
   - For album popularity: JOIN album → track → invoiceline, use SUM(invoiceline.quantity)
   - For artist popularity: JOIN artist → album → track → invoiceline, use SUM(invoiceline.quantity)
   - NEVER use track.quantity (doesn't exist), always use invoiceline.quantity
   
   **Store Patterns:**
   - Product sales: products → orderdetails (SUM quantity * priceEach)
   - Category performance: productlines → products → orderdetails
   - Customer value: customers → orders → orderdetails (SUM revenue)
   - Geographic analysis: customers → orders (GROUP BY country/city)
   - Inventory status: products (quantityInStock analysis)
   - Payment analysis: customers → payments (SUM amounts)
   
   **World Patterns:**
   - Country statistics: country (population, GNP, surface area analysis)
   - Continental analysis: country (GROUP BY Continent)
   - City rankings: city → country (population, regional comparisons)
   - Language distribution: countrylanguage → country (percentage analysis)
   - Capital cities: country JOIN city ON country.Capital = city.ID
   - Regional comparisons: country (GROUP BY Region, Continent)
   
   **IMDB Patterns:**
   - Movie analysis: movie → ratings (rating and popularity analysis)
   - Genre popularity: genre → movie → ratings (ratings by genre)
   - Actor filmography: names → role_mapping → movie (actor's movies)
   - Director filmography: names → director_mapping → movie (director's movies)
   - Box office analysis: movie (worldwide_gross_income analysis)
   - Yearly trends: movie (GROUP BY year)
   - Production analysis: movie (GROUP BY production_company, country)
   - Cast analysis: movie → role_mapping → names (cast size, popular actors)

6. **Table Aliasing and Reference Standards - MANDATORY**
   🚨 **CRITICAL RULE: ONLY use simple single-letter aliases in alphabetical order:**
     * First table: `a` (main table)
     * Second table: `b` (first join)
     * Third table: `c` (second join) 
     * Fourth table: `d` (third join)
     * Fifth table: `e` (fourth join)
     * Sixth table: `f` (fifth join)
   
   🚫 **FORBIDDEN:** Never use aliases like c1, c2, co, ci, sub_c, etc.
   ✅ **REQUIRED:** Always use a, b, c, d, e, f in order
   
   - CRITICAL: Ensure all column references use the correct table alias
   - Example: If country is aliased as 'a', use 'a.Population', not 'country.Population'
   - Always prefix columns with their table alias to avoid ambiguity
   - When self-joining, use distinct aliases like 'a' and 'b' for the same table
   - In subqueries, continue the pattern: use 'c', 'd', 'e', 'f' for subquery tables

7. **Query Structure and Formatting Standards**
   - Write clean SQL with proper indentation and line breaks:
     ```
     SELECT column1, column2, calculation
     FROM table1 AS a
     JOIN table2 AS b ON a.key = b.key
     WHERE condition
     GROUP BY grouping_columns
     HAVING having_condition
     ORDER BY sort_columns
     LIMIT number;
     ```
   - Always use explicit JOIN syntax (JOIN...ON) instead of WHERE clause joins
   - Use appropriate JOIN types:
     * INNER JOIN for required relationships
     * LEFT JOIN when you need all records from the left table
     * RIGHT JOIN when you need all records from the right table
   - For aggregations, ensure all non-aggregate columns are in GROUP BY
   - Use meaningful column names in SELECT, especially for calculations
   - Sort results appropriately (DESC for top/highest, ASC for lowest)

8. **Data Validation and Error Prevention**
   - Before finalizing the query, mentally trace through each table alias
   - Verify that all JOIN conditions match the foreign key relationships in the schema
   - Ensure all column names exactly match the schema (case-sensitive)
   - Check that aggregate functions (SUM, COUNT, AVG) are used appropriately
   - Validate that date columns are handled correctly if time-based filtering is needed
   - For calculations, ensure mathematical operations make logical sense

9. **Output Requirements**
   - Return ONLY the executable SQL query
   - No markdown formatting (no ```sql blocks)
   - No explanatory text or comments
   - No trailing semicolon unless specifically required
   - Ensure the query will execute successfully against the provided schema

10. **Final Validation Checklist**
   - Before returning the query, verify that ALL column references are prefixed with table aliases
   - Check that subqueries use proper table aliases (e.g., sub_c.Population, not Population)
   - Ensure no ambiguous column references exist anywhere in the query
   - Double-check that JOIN conditions use proper table aliases

11. **Detailed Statistics Guidelines**
   - When user asks for "detailed statistics", "comprehensive data", or "all information":
     * Select multiple relevant columns, not just one or two
     * For countries: Include Name, Population, GNP, LifeExpectancy, SurfaceArea, GovernmentForm
     * For cities: Include Name, Population, District, CountryCode
     * For movies: Include title, year, duration, country, languages
     * For music: Include artist, album, track name, genre, duration
   - Always include the most important identifying column (Name, title, etc.)
   - Include quantitative measures (Population, GNP, ratings, sales, etc.)
   - Include qualitative descriptors when relevant (GovernmentForm, Genre, etc.)

12. **CRITICAL: NO DUPLICATE COLUMNS RULE**
   🚨 **NEVER SELECT THE SAME COLUMN TWICE IN A QUERY**
   - WRONG: SELECT a.Name, a.Population, a.Population FROM country a
   - WRONG: SELECT a.Region, a.Region, a.Population FROM country a  
   - CORRECT: SELECT a.Name, a.Population, a.GNP FROM country a
   - For table charts with "detailed data", select 4-6 DIFFERENT columns with DIVERSE information
   - Each column must provide unique, meaningful information
   - Example for countries: Name, Population, GNP, LifeExpectancy, SurfaceArea, Continent
   - Example for customers: FirstName, LastName, Country, City, Email, Phone
   - Example for tracks: Name, Artist, Album, Genre, Duration, Price
   
   🔧 **TABLE-SPECIFIC COLUMN SELECTION:**
   - Choose columns that provide DIFFERENT types of information
   - Mix identifiers, numbers, categories, and dates
   - Avoid selecting the same data with different column names
   - Example for regional data: Region, Year, Population, GDP, LifeExpectancy (NOT Region, Region, Population, Population)

13. **SQL FOCUS INTERPRETATION RULES**
   When sql_focus mentions "Multiple diverse columns":
   - Select at least 4-5 different columns for comprehensive data
   - Include 1 identifier column (Name, Title, ID)
   - Include 2-3 numeric columns (Population, Sales, Rating, Price)
   - Include 1-2 categorical columns (Country, Genre, Category)
   - NEVER repeat any column in the SELECT clause
   
   Examples:
   - Countries: Name, Population, GNP, LifeExpectancy, SurfaceArea
   - Movies: title, year, duration, avg_rating, total_votes
   - Customers: FirstName, LastName, Country, City, SupportRepId
   - Tracks: Name, Composer, Milliseconds, UnitPrice, GenreId

14. **CRITICAL: LARGEST PER GROUP QUERIES**
   For questions like "largest cities by continent" or "top X per category":
   - Use window functions or correlated subqueries for proper grouping
   - NEVER use simple GROUP BY with MAX() for non-aggregate columns
   - For largest city per continent: Use ROW_NUMBER() OVER (PARTITION BY continent ORDER BY population DESC)
   - Always ensure the result shows ONE representative per group
   
   Example for "largest city per continent":
   ```sql
   SELECT a.Continent, b.Name, b.Population, b.District
   FROM country a 
   JOIN city b ON a.Code = b.CountryCode
   WHERE b.Population = (
     SELECT MAX(c.Population) 
     FROM city c 
     JOIN country d ON c.CountryCode = d.Code 
     WHERE d.Continent = a.Continent
   )
   ```

Schema:
{schema}

Question:
{question}

SQL Query:
"""
)




# Response prompt for formatting results
response_prompt = ChatPromptTemplate.from_template(
    """You are an assistant that formats SQL query results into a JSON list suitable for visualization.

Schema:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
Chart Type: {chart_type}

IMPORTANT: The SQL response is a string representation of data. Parse it carefully:
- If it looks like: [('Item1', 123), ('Item2', 456)] - this is a list of tuples
- If it looks like: [['Item1', 123], ['Item2', 456]] - this is a list of lists
- If it looks like: [('Item1', 123, 456, 'Text'), ('Item2', 789, 101, 'Text2')] - this has multiple columns
- Extract the actual data values, not the string representation

For detailed statistics with multiple columns:
- Use the first column as the label (usually Name, title, etc.)
- Use the second column as the primary value (usually Population, count, etc.)
- For tables with multiple columns, include all relevant columns in the data objects

Output ONLY in the following JSON format (use double braces for literal curly braces):

For simple data (2 columns):
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "{chart_type}",
  "data": [
    {{ "label": "Label 1", "value": 123 }},
    {{ "label": "Label 2", "value": 456 }}
  ]
}}

For detailed statistics (multiple columns):
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "{chart_type}",
  "data": [
    {{ "label": "Country 1", "value": 123, "gnp": 456, "life_expectancy": 78.5 }},
    {{ "label": "Country 2", "value": 789, "gnp": 101, "life_expectancy": 82.1 }}
  ]
}}

For scatter plots:
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "scatter",
  "data": [
    {{ "label": "Point 1", "x": 123, "y": 456 }},
    {{ "label": "Point 2", "x": 789, "y": 101 }}
  ]
}}

For heatmaps:
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "heatmap",
  "data": [
    {{ "x": "Category A", "y": "Subcategory 1", "value": 123 }},
    {{ "x": "Category A", "y": "Subcategory 2", "value": 456 }},
    {{ "x": "Category B", "y": "Subcategory 1", "value": 789 }}
  ]
}}

For histograms:
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "histogram",
  "data": [
    {{ "bin": "0-10", "value": 15 }},
    {{ "bin": "10-20", "value": 23 }},
    {{ "bin": "20-30", "value": 18 }}
  ]
}}

For stacked bar charts:
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "stacked_bar",
  "data": [
    {{ "label": "Category 1", "series1": 100, "series2": 50, "series3": 25 }},
    {{ "label": "Category 2", "series1": 80, "series2": 60, "series3": 40 }}
  ]
}}

For area charts:
{{
  "title": "<descriptive chart title>",
  "x_axis": "<x-axis label>",
  "y_axis": "<y-axis label>", 
  "chart_type": "area",
  "data": [
    {{ "label": "2020", "value": 100 }},
    {{ "label": "2021", "value": 120 }},
    {{ "label": "2022", "value": 150 }}
  ]
}}

CRITICAL JSON RULES:
- Return ONLY the JSON object, nothing else
- No explanations, no notes, no additional text
- No markdown code blocks (no ```json)
- No trailing text after the closing brace
- Ensure all string values are properly quoted
- Use actual data from the SQL result, not placeholders
- Parse the SQL response string to extract real data values

Important formatting rules:
- For pie charts: Ensure values are percentages or proportions that add up meaningfully
- For line charts: Labels should be sequential (years, months, dates, etc.)
- For scatter plots: Include both x and y values as {{ "label": "Item", "x": 123, "y": 456 }}
- For tables: Use "table" as chart_type and structure data appropriately
- For heatmaps: Use x, y, and value properties for 2D data visualization
- For histograms: Use bin ranges and frequency values for distribution analysis
- For stacked bar charts: Include multiple series values for layered comparisons
- For area charts: Use sequential labels with cumulative or filled area data
- Always provide descriptive, clear labels and titles

Critical SQL Rules:
When generating SQL queries, you must fully comply with MySQL's ONLY_FULL_GROUP_BY mode:
- Every column in SELECT that is not inside an aggregate (SUM, MAX, COUNT, etc.) must appear in GROUP BY
- If you want a non-aggregated column (e.g., city name with MAX population), use a subquery or window function
- Never select non-aggregated columns alongside aggregates without fixing GROUP BY
- When asked for "largest per group" (e.g., largest city per continent), never mix aggregate functions with non-aggregated columns in the SELECT clause. Instead, use either a correlated subquery (classic MySQL) or a window function (ROW_NUMBER() or RANK()) to ensure the non-aggregated column matches the aggregate.
- Never write a query that selects a non-aggregated column with an aggregate (like MAX, SUM, etc.) without ensuring the non-aggregated column is included in GROUP BY or properly correlated with a subquery/window function. For cases like "largest city per country", always use a correlated subquery or a window function instead of GROUP BY

AMBIGUOUS COLUMN RULE (CRITICAL):
- ALWAYS prefix column names with their table alias to avoid ambiguous column errors
- When joining tables that have columns with the same name (e.g., Population in both country and city tables), you MUST specify which table's column you want: ci.Population for city population, c.Population for country population
- Example: Use "ci.Population" not "Population" when both country and city tables are joined
- This prevents "Column 'X' in field list is ambiguous" errors
- CRITICAL: In subqueries, also prefix ALL column references with table aliases
- Example: Use "sub_c.Population" not "Population" in subqueries

COMMON AMBIGUOUS COLUMN PATTERNS TO AVOID:
- WRONG: SELECT MAX(Population) FROM city JOIN country ON city.CountryCode = country.Code
- CORRECT: SELECT MAX(city.Population) FROM city JOIN country ON city.CountryCode = country.Code
- WRONG: WHERE city.Population = (SELECT MAX(Population) FROM city JOIN country WHERE country.Continent = co.Continent)
- CORRECT: WHERE a.Population = (SELECT MAX(c.Population) FROM city c JOIN country d ON c.CountryCode = d.Code WHERE d.Continent = b.Continent)

LARGEST PER GROUP RULE (CRITICAL):
- For queries like "largest city per continent" or "largest X per Y", NEVER use GROUP BY with non-aggregated columns
- Instead, use a correlated subquery or window function approach:
  * CORRELATED SUBQUERY: WHERE a.Population = (SELECT MAX(c.Population) FROM city c JOIN country d ON c.CountryCode = d.Code WHERE d.Continent = b.Continent)
  * WINDOW FUNCTION: Use ROW_NUMBER() OVER (PARTITION BY c.Continent ORDER BY ci.Population DESC) and filter WHERE rn = 1
- Example for "largest city per continent":
  ```sql
  SELECT a.Continent, b.Name AS LargestCity, b.Population
  FROM country a 
  JOIN city b ON a.Code = b.CountryCode
  WHERE b.Population = (
    SELECT MAX(c.Population) 
    FROM city c 
    JOIN country d ON c.CountryCode = d.Code 
    WHERE d.Continent = a.Continent
  )
  ```
- CRITICAL: Always use simple a,b,c,d,e,f aliases in subqueries to avoid ambiguous column errors
- WRONG: SELECT MAX(Population) FROM city (ambiguous - which Population?)
- CORRECT: SELECT MAX(c.Population) FROM city c (clear - city population with simple alias)
""")
# Function to create SQL chain for specific database
def create_sql_chain(database_name="chinook"):
    return (
        RunnablePassthrough.assign(schema=lambda x: get_schema(database_name))
        | sql_prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

# Helper function to safely convert values to float
def safe_float(value):
    """Convert various numeric types to float safely"""
    try:
        if hasattr(value, '__float__'):
            return float(value)
        return float(str(value))
    except (ValueError, TypeError):
        return 0.0

def generate_axis_labels(chart_type, columns, question, title):
    """Generate intelligent axis labels based on context"""
    if not columns or len(columns) < 2:
        return "Categories", "Values"
    
    # Clean column names
    clean_columns = [col.replace('a.', '').replace('b.', '').replace('c.', '').replace('d.', '').replace('e.', '').replace('f.', '') for col in columns]
    
    # Special logic for scatter plots - find meaningful numeric columns
    if chart_type == "scatter":
        # For scatter plots, we need to find the best numeric columns for X and Y
        # Based on the format_data_for_chart_type logic, scatter uses columns 1 and 2 as x,y
        if len(clean_columns) >= 3:
            x_col = clean_columns[1]  # Second column (Population in your example)
            y_col = clean_columns[2]  # Third column (GNP in your example)
        else:
            x_col = clean_columns[0] if len(clean_columns) > 0 else "X"
            y_col = clean_columns[1] if len(clean_columns) > 1 else "Y"
        
        x_axis = generate_readable_label(x_col, "x", question)
        y_axis = generate_readable_label(y_col, "y", question)
        
        # Add units/context for common numeric columns
        if "population" in x_col.lower():
            x_axis = f"{x_axis}"
        if "gnp" in y_col.lower():
            y_axis = f"{y_axis} (in millions USD)"
        if "lifeexpectancy" in y_col.lower():
            y_axis = f"{y_axis} (years)"
        if "surfacearea" in y_col.lower():
            y_axis = f"{y_axis} (km²)"
            
        return x_axis, y_axis
    
    # For other chart types, use first and last columns
    x_col = clean_columns[0]
    y_col = clean_columns[-1]  # Last column is usually the aggregated value
    
    # Generate meaningful axis labels based on column names and context
    x_axis = generate_readable_label(x_col, "x", question)
    y_axis = generate_readable_label(y_col, "y", question)
    
    # Chart-specific adjustments for non-scatter charts
    if chart_type in ["bar", "line"]:
        # For bar/line charts, make sure Y-axis indicates it's a measurement
        if not any(word in y_axis.lower() for word in ["total", "sum", "count", "average", "amount", "number"]):
            if "population" in y_col.lower():
                y_axis = f"Total {y_axis}"
            elif "purchase" in y_col.lower() or "sales" in y_col.lower():
                y_axis = f"Total {y_axis}"
            elif "gnp" in y_col.lower():
                y_axis = f"{y_axis} (in millions USD)"
    
    return x_axis, y_axis

def generate_readable_label(column_name, axis_type, question):
    """Convert database column names to readable labels"""
    if not column_name:
        return "Categories" if axis_type == "x" else "Values"
    
    # Handle common column name patterns
    column_lower = column_name.lower()
    
    # Common mappings
    label_mappings = {
        'customerid': 'Customer ID',
        'firstname': 'First Name', 
        'lastname': 'Last Name',
        'totalpurchases': 'Total Purchases ($)',
        'population': 'Population',
        'gnp': 'GNP (Gross National Product)',
        'lifeexpectancy': 'Life Expectancy (Years)',
        'surfacearea': 'Surface Area (km²)',
        'region': 'Region',
        'continent': 'Continent',
        'country': 'Country',
        'countrycode': 'Country Code',
        'name': 'Name',
        'city': 'City',
        'district': 'District',
        'language': 'Language',
        'percentage': 'Percentage (%)',
        'isofficial': 'Official Language',
        'indepyear': 'Independence Year',
        'headofstate': 'Head of State',
        'governmentform': 'Government Form',
        'code': 'Code',
        'code2': 'Country Code (2-letter)',
        'capital': 'Capital City',
        'email': 'Email Address',
        'phone': 'Phone Number',
        'address': 'Address',
        'company': 'Company',
        'state': 'State/Province',
        'postalcode': 'Postal Code',
        'fax': 'Fax Number'
    }
    
    # Check for exact matches first
    if column_lower in label_mappings:
        return label_mappings[column_lower]
    
    # Check for partial matches
    for key, value in label_mappings.items():
        if key in column_lower:
            return value
    
    # If no mapping found, convert camelCase/snake_case to readable format
    import re
    
    # Convert camelCase to words
    readable = re.sub(r'([a-z])([A-Z])', r'\1 \2', column_name)
    # Convert snake_case to words  
    readable = readable.replace('_', ' ')
    # Capitalize each word
    readable = ' '.join(word.capitalize() for word in readable.split())
    
    return readable

def generate_narrative(question: str, charts: list) -> dict:
    """Generate connecting narrative for charts (single or multiple)"""
    try:
        # Create chart info summary for the AI
        chart_info = []
        for i, chart in enumerate(charts, 1):
            chart_info.append(f"Chart {i}: {chart.get('chart_type', 'unknown')} - {chart.get('title', 'Untitled')}")
        
        chart_info_str = "\n".join(chart_info)
        print(f"Chart info for narrative: {chart_info_str}")
        
        # Generate narrative
        chain = narrative_prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "chart_info": chart_info_str
        })
        
        print(f"Raw narrative response: {response}")
        
        # Clean up response (remove markdown if present)
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        print(f"Cleaned narrative response: {cleaned_response}")
        
        # Parse JSON
        narrative = json.loads(cleaned_response)
        print(f"Parsed narrative: {narrative}")
        return narrative
        
    except Exception as e:
        print(f"Error generating narrative: {e}")
        # Return fallback narrative based on number of charts
        if len(charts) == 1:
            return {
                "introduction": f"Let's examine the data about {question.lower()}.",
                "transitions": [],  # No transitions needed for single chart
                "insights": [
                    "This visualization reveals key patterns in the data.",
                    "The chart provides clear insights into the underlying trends."
                ],
                "conclusion": "This analysis helps us understand the important aspects of the data."
            }
        else:
            return {
                "introduction": f"Let's explore the data about {question.lower()}.",
                "transitions": ["The data reveals interesting patterns as we examine different perspectives."] * max(0, len(charts) - 1),
                "insights": ["The data shows significant variations across different categories."],
                "conclusion": "These visualizations provide valuable insights into the underlying patterns."
            }

# Helper function to format data for specific chart types
def format_data_for_chart_type(data, chart_type, question, columns=None):
    """
    Format data appropriately for different chart types.
    Automatically limits large datasets for better visualization.
    """
    if not data:
        return []
    
    # Limit data size for better visualization and performance
    original_length = len(data)
    max_items = {
        'pie': 6,      # Pie charts should have very few slices
        'bar': 20,     # Bar charts can handle more items
        'line': 50,    # Line charts can show more data points
        'scatter': 100, # Scatter plots can handle many points
        'table': 50    # Tables can show more rows but limit for performance
    }
    
    limit = max_items.get(chart_type, 20)  # Default to 20 items
    
    if original_length > limit:
        print(f"📊 Data too large ({original_length} items), limiting to top {limit} for {chart_type} chart")
        data = data[:limit]  # Take first N items (assuming data is already sorted by importance)
    
    if chart_type == "scatter":
        # For scatter plots, ensure we have x and y values
        formatted_data = []
        for item in data:
            if len(item) >= 3:  # label, x, y
                formatted_data.append({
                    "label": str(item[0]),
                    "x": safe_float(item[1]),
                    "y": safe_float(item[2]),
                    "value": 0
                })
        return formatted_data
    
    
    else:
        # Default formatting for bar, line, pie, table, area
        formatted_data = []
        for item in data:
            if len(item) >= 2:
                # Find the numeric value column (usually the last column for aggregated data)
                numeric_value = 0.0
                
                # Look for a numeric column, preferring the last column
                for i in range(len(item) - 1, 0, -1):  # Start from last, go backwards
                    try:
                        numeric_value = safe_float(item[i])
                        if numeric_value != 0.0:  # Found a non-zero numeric value
                            break
                    except:
                        continue
                
                # If still 0, try item[1] as fallback
                if numeric_value == 0.0:
                    numeric_value = safe_float(item[1])
                
                # Smart label assignment - prefer country/city names over continents
                label_value = str(item[0])  # Default to first column
                
                # If we have columns info, try to find a better label
                if columns and len(columns) > 1:
                    # Look for country name, city name, or other meaningful identifiers
                    for i, col in enumerate(columns):
                        if i < len(item):
                            col_lower = col.lower().replace('a.', '').replace('b.', '').replace('c.', '')
                            # Prefer country names, city names over continents
                            if 'countryname' in col_lower or 'name' in col_lower and 'continent' not in col_lower:
                                if str(item[i]) and str(item[i]) != 'None':
                                    label_value = str(item[i])
                                    break
                
                data_obj = {"label": label_value, "value": numeric_value}
                
                # Add additional columns with proper names and avoid duplicates
                if columns and len(columns) > 1:
                    added_columns = set(['label', 'value'])  # Track what we've already added
                    
                    for i, val in enumerate(item):
                        if i < len(columns):
                            col_name = columns[i]
                            # Clean up column names
                            col_name = col_name.replace('a.', '').replace('b.', '').replace('c.', '')
                            original_col_name = col_name.lower()
                            
                            # Skip if this column represents the same data as label or value
                            if (original_col_name in ['region', 'name', 'title'] and 
                                str(val) == data_obj['label']):
                                continue
                                
                            if (original_col_name in ['population', 'total', 'amount', 'count'] and 
                                safe_float(val) == data_obj['value']):
                                continue
                            
                            # Avoid duplicate column names
                            if col_name.lower() in added_columns:
                                col_name = f"{col_name}_{i}"
                            
                            data_obj[col_name] = val
                            added_columns.add(col_name.lower())
    else:
                    # Fallback to generic column names
                for i, val in enumerate(item[2:], 2):
                    data_obj[f"col_{i}"] = val
                formatted_data.append(data_obj)
        return formatted_data



# Function to create full chain for specific database with intelligent chart selection
def create_multiple_charts(question: str, database_name="chinook"):
    """Generate multiple charts for a single question"""
    try:
        # Get chart suggestions using the prompt directly
        try:
            schema = get_schema(database_name)
            chain = chart_suggestion_prompt | llm | StrOutputParser()
            response = chain.invoke({"schema": schema, "question": question})
            
            # Parse the JSON response
            try:
                # Clean up the response - remove markdown code blocks
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                print(f"Cleaned response: {cleaned_response[:200]}...")
                
                suggestions_data = json.loads(cleaned_response)
                suggestions = suggestions_data.get("suggestions", [])
                print(f"AI suggested {len(suggestions)} charts:")
                for i, suggestion in enumerate(suggestions):
                    print(f"  {i+1}. {suggestion.get('chart_type')} - {suggestion.get('title')}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse chart suggestions JSON: {e}")
                print(f"Raw response: {response}")
                # Fallback if JSON parsing fails
                suggestions = [
                    {
                        "chart_type": "bar",
                        "title": f"Analysis: {question[:50]}...",
                        "reason": "Bar chart for data comparison",
                        "sql_focus": "Main data points"
                    }
                ]
        except Exception as e:
            print(f"Error in chart suggestion: {e}")
            # Fallback suggestions
            suggestions = [
                {
                    "chart_type": "bar",
                    "title": f"Analysis: {question[:50]}...",
                    "reason": "Default bar chart",
                    "sql_focus": "Main data points"
                }
            ]
        
        if not suggestions:
            # Fallback to single chart
            return create_single_chart(question, database_name)
        
        charts = []
        sql_chain = create_sql_chain(database_name)
        
        for suggestion in suggestions:
            try:
                chart_type = suggestion.get("chart_type", "bar")
                title = suggestion.get("title", f"Analysis: {question[:50]}...")
                sql_focus = suggestion.get("sql_focus", "Main data points")
                
                # Create modified inputs with chart focus
                modified_inputs = {
                    "question": f"{question} - Focus: {sql_focus}",
                    "database": database_name
                }
                
                # Generate SQL query
                query = sql_chain.invoke(modified_inputs)
                print(f"Generated SQL for {chart_type}: {query}")
                response, columns = run_query_with_columns(query, database_name)
                print(f"SQL Response for {chart_type}: {response}")
                print(f"Columns for {chart_type}: {columns}")
                
                # Parse and format data using robust parsing logic
                try:
                    print(f"Response type: {type(response)}")
                    print(f"Response is list: {isinstance(response, list)}")
                    
                    # Response from run_query_with_columns is always a list
                    parsed_data = response
                    print(f"Using list response directly: {len(parsed_data)} rows")

    except Exception as e:
                    print(f"Error parsing data for {chart_type}: {e}")
                    parsed_data = []
                
                print(f"Parsed data for {chart_type}: {parsed_data}")
                
                # Format data for the specific chart type with column names
                formatted_data = format_data_for_chart_type(parsed_data, chart_type, question, columns)
                print(f"Formatted data for {chart_type}: {formatted_data}")
                
                # Add note if data was limited
                original_count = len(parsed_data) if parsed_data else 0
                final_count = len(formatted_data) if formatted_data else 0
                if original_count > final_count and final_count > 0:
                    title += f" (Top {final_count})"
                
                # Create chart data with intelligent axis labels
                x_axis, y_axis = generate_axis_labels(chart_type, columns, question, title)
                
                chart_data = {
                    "title": title,
                    "x_axis": x_axis,
                    "y_axis": y_axis,
                    "chart_type": chart_type,
                    "data": formatted_data
                }
                
                charts.append(chart_data)
                
            except Exception as e:
                print(f"Error creating chart for {suggestion.get('chart_type', 'unknown')}: {e}")
                # Add error chart
                charts.append({
                    "title": f"Error: {suggestion.get('title', 'Chart')}",
                    "x_axis": "Error",
                    "y_axis": "Count",
                    "chart_type": "bar",
                    "data": [{"label": "Error", "value": 1}]
                })
        
        # Generate narrative for charts (both single and multiple)
        print(f"Total charts generated: {len(charts)}")
        if len(charts) >= 1:
            print(f"Generating narrative for {len(charts)} chart(s)...")
            narrative = generate_narrative(question, charts)
            print(f"Generated narrative: {narrative}")
            
            # Return charts with narrative (even for single chart)
            result = {
                "charts": charts,
                "narrative": narrative
            }
            print(f"Returning result with narrative: {result.keys()}")
            return result
        else:
            print("No charts generated, returning None")
            return None
        
    except Exception as e:
        print(f"Error in create_multiple_charts: {e}")
        return create_single_chart(question, database_name)

def create_single_chart(question: str, database_name="chinook"):
    """Create a single chart (original functionality)"""
    sql_chain = create_sql_chain(database_name)
    
    def process_with_chart_type(inputs):
        # First, get the SQL query and run it
        query = sql_chain.invoke(inputs)
        schema = get_schema(database_name)
        response, columns = run_query_with_columns(query, database_name)
        
        # Get chart suggestions and use the first one for single chart
        try:
            chain = chart_suggestion_prompt | llm | StrOutputParser()
            suggestion_response = chain.invoke({"schema": schema, "question": inputs["question"]})
            
            suggestions_data = json.loads(suggestion_response)
            suggestions = suggestions_data.get("suggestions", [])
            
            if suggestions:
                chart_type = suggestions[0].get("chart_type", "bar")
            else:
                chart_type = "bar"
        except Exception as e:
            print(f"Error getting chart suggestion: {e}")
            chart_type = "bar"
        
        # Parse the SQL response into structured data
        try:
            # Response from run_query_with_columns is already a list
            parsed_data = response
            
            # Format data for the specific chart type with column names
            formatted_data = format_data_for_chart_type(parsed_data, chart_type, inputs["question"], columns)
            
        except Exception as e:
            print(f"Error parsing data: {e}")
            print(f"Response type: {type(response)}")
            print(f"Response content: {response}")
            # Fallback to simple chart creation
            chart_json = {
                "title": f"Analysis: {inputs['question'][:50]}...",
                "x_axis": "Categories" if chart_type in ["bar", "pie", "histogram"] else "X Values",
                "y_axis": "Values" if chart_type in ["bar", "line", "pie", "histogram", "area"] else "Y Values",
                "chart_type": chart_type,
                "data": [{"label": "Error", "value": 1}]
            }
            return chart_json
        
        # Create chart JSON with intelligent axis labels
        title = f"Analysis: {inputs['question'][:50]}..."
        x_axis, y_axis = generate_axis_labels(chart_type, columns, inputs["question"], title)
        
        chart_json = {
            "title": f"Analysis: {inputs['question'][:50]}...",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "chart_type": chart_type,
            "data": formatted_data
        }
        
        return chart_json
    
    return process_with_chart_type

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Backend is running"})

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Process natural language question and return chart data"""
    # Initialize output_str to avoid UnboundLocalError
    output_str = ""
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        database = data.get('database', 'chinook')  # Default to chinook
        
        if database not in databases:
            return jsonify({"error": f"Database '{database}' not found"}), 400
        
        # Check if user wants multiple charts (default to True for now)
        generate_multiple = data.get('multiple_charts', True)
        
        if generate_multiple:
            # Generate multiple charts
            result = create_multiple_charts(question, database)
            
            # Check if result contains narrative (multiple charts) or is a single chart
            if isinstance(result, dict) and "charts" in result and "narrative" in result:
                # Multiple charts with narrative
                return jsonify({
                    "success": True,
                    "data": result["charts"],
                    "narrative": result["narrative"],
                    "question": question,
                    "database": database
                })
            else:
                # Single chart or fallback
                return jsonify({
                    "success": True,
                    "data": result,
                    "question": question,
                    "database": database
                })
        else:
            # Generate single chart (original behavior)
            chart_data = create_single_chart(question, database)
        
        return jsonify({
            "success": True,
            "data": chart_data,
            "question": question,
            "database": database
        })
            
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Failed to parse chart data",
            "details": str(e)
        }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/execute-sql', methods=['POST'])
def execute_sql():
    """Execute raw SQL query (for debugging)"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "SQL query is required"}), 400
        
        query = data['query']
        database = data.get('database', 'chinook')
        
        if database not in databases:
            return jsonify({"error": f"Database '{database}' not found"}), 400
        
        result = run_query(query, database)
        
        return jsonify({
            "success": True,
            "result": result,
            "query": query,
            "database": database
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/databases', methods=['GET'])
def get_databases():
    """Get list of available databases"""
    try:
        databases_info = get_available_databases()
        return jsonify({
            "success": True,
            "databases": databases_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/schema', methods=['GET'])
def get_database_schema():
    """Get database schema information"""
    try:
        database = request.args.get('database', 'chinook')
        
        if database not in databases:
            return jsonify({"error": f"Database '{database}' not found"}), 400
        
        schema = get_schema(database)
        return jsonify({
            "success": True,
            "schema": schema,
            "database": database
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
