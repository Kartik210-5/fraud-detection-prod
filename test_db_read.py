import os
import pandas as pd
from supabase import create_client

url = "https://mqdbfqlhjddqqncyrwpy.supabase.co"
key = "sb_publishable_vb9pr-pn42rp3UOhGtYRAQ_woFkX5dg"

try:
    sb = create_client(url, key)
    res = sb.table("predictions").select("*").limit(5).execute()
    if res.data:
        print(f"✅ Found {len(res.data)} rows in Supabase table!")
        print(pd.DataFrame(res.data))
    else:
        print("Empty Table: Connection works, but table contains 0 rows.")
except Exception as e:
    print(f"❌ Connection Error: {e}")