import csv
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from llm_handler import send_to_llm
from params import OUTPUT_FILE_PATH, NUM_WORKERS, PROVIDER, OUTPUT_CSV_PATH
from system_messages import SYSTEM_MESSAGES_GENERAL
from topics import TOPICS

def generate_data(topic_selected, system_message_selected):
    # Construct the message list based on the new structure
    msg_list = [
        {"role": "system", "content": system_message_selected},
        {"role": "user", "content": f"SUBJECT_AREA: {topic_selected}"}
    ]
    
    # Send message list to the LLM
    llm_response, _ = send_to_llm(PROVIDER, msg_list)
    
    # Prepare the data for writing
    data = {
        "system": system_message_selected,
        "topic": topic_selected,
        "response": llm_response
    }
    
    return data

def main():
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor, \
         open(OUTPUT_FILE_PATH, "a") as jsonl_file, \
         open(OUTPUT_CSV_PATH, 'a', newline='') as csv_file:
         
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['System Message', 'Topic', 'Response'])  # Write CSV headers

        futures = []
        for _ in range(NUM_WORKERS):
            topic_number = np.random.randint(0, len(TOPICS))
            topic_selected = TOPICS[topic_number]
            system_message_number = np.random.randint(0, len(SYSTEM_MESSAGES_GENERAL))
            system_message_selected = SYSTEM_MESSAGES_GENERAL[system_message_number]
            
            futures.append(
                executor.submit(
                    generate_data,
                    topic_selected,
                    system_message_selected
                )
            )

        for future in futures:
            data = future.result()
            jsonl_file.write(json.dumps(data) + "\n")  # Write JSON Line
            csv_writer.writerow([data['system'], data['topic'], data['response']])  # Write CSV row

            print(data)  # Optional: print data to console

if __name__ == "__main__":
    while True:
        main()
