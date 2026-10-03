from agent.cfo_agent import run_cfo_analysis


sample_company_data = {
    "revenue": 1000000,
    "operating_cash_flow": 120000,
    "ebitda": 180000,
    "cash_ratio": 0.15
}


result = run_cfo_analysis(sample_company_data)

print(result)
