import os
import pandas as pd
from groq import Groq
import matplotlib.pyplot as plt
import traceback

#====================
# Config
#====================
groq_client = (
    Groq(api_key=os.getenv("groq_api_key")))

#====================
# Load Excel Data
#====================
def load_excel(filename: str) \
        -> pd.DataFrame:
    df = pd.DataFrame()
    try:
        df = pd.read_excel(filename)
        print("Excel Data Loaded in "
              "DataFrame -> "
              ""+ str(df.head(5)))
    except Exception as e:
        print("Error while Loading Excel"
              + str(e))
    return df

#====================
# Generate Context
#====================
def generate_context(df: pd.DataFrame) \
        -> dict:

    columns = list(df.columns)
    column_datatypes = (df.dtypes.astype(str).
                        to_dict())
    size = len(df)
    product_per_region_sales = (
        (df.groupby(["region","product"]).
        agg(row_count=("product", "count"))).
        reset_index())
    total_revenue_per_region = \
        ((df.groupby(["region"])).
        agg(total_sales=("revenue", "sum")).
        reset_index())
    total_revenue_per_product = \
        ((df.groupby(["product"])).
        agg(total_sales=("revenue", "sum")).
        reset_index())
    quantity_sold_per_region_product = \
        ((df.groupby(["region","product"])).
        agg(total_sales=("quantity", "sum")).
        reset_index())
    category_per_region_sales = \
        ((df.groupby(["region","category"])).
        agg(row_count=("category", "count")).
        reset_index())
    total_revenue_per_category = \
        ((df.groupby(["category"])).
        agg(total_sales=("revenue", "sum")).
        reset_index())

    return {
        "columns": columns,
        "column_datatypes": column_datatypes,
        "number_of_rows": size,
        "product_per_region_sales":
            product_per_region_sales.
            to_dict(orient="records"),
        "total_revenue_per_region":
            total_revenue_per_region.
            to_dict(orient="records"),
        "total_revenue_per_product":
            total_revenue_per_product.
            to_dict(orient="records"),
        "quantity_sold_per_region_product":
            quantity_sold_per_region_product.
            to_dict(orient="records"),
        "category_per_region":
            category_per_region_sales.
            to_dict(orient="records"),
        "total_revenue_per_category":
            total_revenue_per_category.
            to_dict(orient="records")
    }

#====================
# Generate Chart Code
#====================
def generate_code(context: dict, client):
    message = f"""
    You are a Senior Python Developer and Data 
    Visualization Expert.

You are given the following dataset context:
{context}

Your task is to generate clean, 
production-quality Python code to create 
meaningful and insightful data visualizations.

Requirements:
- Remove all comments from python code.
- Use only the matplotlib library for 
generating charts and graphs.
- The code must be fully executable as-is.
- Assume the DataFrame is named `df`.
- Do not include any explanations, comments, 
or markdown—output only Python code.
- Ensure proper handling of data types 
where necessary (e.g., numeric conversions).
- Generate only those visualizations that are 
relevant, 
meaningful, and add analytical value 
based on the dataset context.
- Avoid redundant or trivial charts.
- Generate bar chart, pie chart, 
histogram and line chart for all types of data.
- Use subplots in case of multiple columns.
- Ensure charts are properly 
labeled (title, axes) for clarity.

Output strictly valid Python code 
and nothing else.
    """
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[  # type: ignore
            {"role": "system",
             "content": "You generate correct "
                        "pandas and matplotlib "
                        "code."},
            {"role": "user", "content": message}
        ]
    )
    return (
        str(response.choices[0].message.
            content.strip()).
        replace("```", ""))[6:]

#====================
# Execute Graph Code
#====================
def execute_code(code : str):
    try:
        exec(code)
        print("Python Code to Generate Graph "
              "Executed Successfully")
    except Exception as e:
        print("Error while Executing Python "
              "Code to Generate "
              "Graph -> "+ str(e))

#====================
# Generate Insights
#====================
def generate_insights(context: dict,
                      client: groq_client):

    message = f"""You are a senior data analyst.

Dataset:
{context}

Explain insights in simple bullet points.
Focus on business insights.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[  # type: ignore
            {"role": "system",
             "content": "You generate simple "
                        "business insight "
                        "based on provided "
                        "dataset."},
            {"role": "user", "content": message}
        ]
    )
    return str(response.choices[0].
               message.content.
               strip())

#====================
# Main Pipeline Steps
#====================

# 1. Load Excel Data
df_excel = load_excel("sample.xlsx")

# 2. Build Context
context = generate_context(df_excel)

# 3. Generate Chart Python Code
python_code = (
    generate_code(context, groq_client))
print(python_code)

# 4. Execute Chart Python Code
execute_code(python_code)

# 5. Generate Business Insight
insights = (
    generate_insights(context, groq_client))
print(insights)
