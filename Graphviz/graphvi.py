import pandas as pd
from graphviz import Digraph

# File paths
leads_file = "Data Kaggle\leads_basic_details.csv"
managers_file = "Data Kaggle\sales_managers_assigned_leads_details.csv"

# Load CSV data
leads_df = pd.read_csv(leads_file)
managers_df = pd.read_csv(managers_file)



# Extract unique Senior and Junior Sales Managers
senior_managers = managers_df["snr_sm_id"].unique()
junior_managers = managers_df["jnr_sm_id"].unique()

# Create a dictionary mapping Junior Managers to their assigned Leads
junior_to_leads = managers_df.groupby("jnr_sm_id")["lead_id"].apply(list).to_dict()

# Create a dictionary mapping Senior Managers to their Junior Managers
senior_to_junior = managers_df.groupby("snr_sm_id")["jnr_sm_id"].unique().to_dict()

# Create the Sales Hierarchy Graph
hierarchy = Digraph("SalesHierarchy")
hierarchy.attr(rankdir="TB")  # Top to Bottom layout

# Add Senior Sales Managers (Purple)
for sm in senior_managers:
    hierarchy.node(sm, shape="box", style="filled", fillcolor="purple", fontcolor="white")

# Add Junior Sales Managers (Blue) and connect them to Senior Managers
for sm, juniors in senior_to_junior.items():
    for jm in juniors:
        hierarchy.node(jm, shape="box", style="filled", fillcolor="lightblue")
        hierarchy.edge(sm, jm)

# Add Leads (Pink) and connect them to Junior Sales Managers
for jm, leads in junior_to_leads.items():
    for lead in leads:
        hierarchy.node(lead, shape="ellipse", style="filled", fillcolor="pink")
        hierarchy.edge(jm, lead)

# Save and render
# hierarchy.render("/mnt/data/sales_hierarchy", format="png", view=False)
# "/mnt/data/sales_hierarchy.png"  # Return the file path for access

hierarchy.engine = "sfdp"  # Alternative: "neato", "fdp", "twopi"
hierarchy.render("/mnt/data/sales_hierarchy", format="png", view=True)
