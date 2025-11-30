prepare-data:
	python walmart_ahmedkobtan_agentic_store_operations/src/utils/prepare_uci_online_retail.py

check-roster:
	python makefile_scripts/check_roster.py

propose:
	python walmart_ahmedkobtan_agentic_store_operations/src/services/propose_schedule.py

evaluate:
	python walmart_ahmedkobtan_agentic_store_operations/src/services/evaluate_schedule.py

ui:
	streamlit run walmart_ahmedkobtan_agentic_store_operations/src/app/main.py

test:
	pytest -q
