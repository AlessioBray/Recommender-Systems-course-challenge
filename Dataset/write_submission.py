"""
Created on 06/11/21
@author: Alessio Bray
"""

import os

def write_submission(recommended_items, filename = 'submission.csv'):
    
    DIRECTORY_PATH = "./Submissions"
    file_path = os.path.join(DIRECTORY_PATH, filename)
    
    if not os.path.exists(DIRECTORY_PATH):
        os.makedirs(DIRECTORY_PATH)
    
    with open(file_path, "w") as f:
        f.write(f"{'user_id'},{'item_list'}\n")
        for user_id, items in enumerate(recommended_items):
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")