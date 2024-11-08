# app.py
from flask import Flask, render_template, request
import csv
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from annoy import AnnoyIndex
import cohere
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
from flask import Flask, request, jsonify, render_template
warnings.filterwarnings('ignore')



app = Flask(__name__)

df =  pd.read_excel("ucdp-peace-agreements-221.xlsx")


code_input = "Issue_groups.csv"
input_file = "issues.csv"
column_index = 16
output_file = "issue_column.csv"
dyad_id = []
locations = []
conflicting_parties_1 = []
conflicting_parties_2 = []
location = []
side_b = []
year = []
issue_text = []
count = 0
issue_level = ["Level 1", "Level 2", "Level 3"]



with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
    reader = csv.reader(csvfile,delimiter=';')
    for row in reader: 
        count = count + 1; 
        
        if row[11] == '1000':
            continue;
        
        if row[4] not in locations:
            locations.append(row[4])
      
def CreateIndex(loaded_embeddings , method):
    embeds = np.array(loaded_embeddings)
    search_index = AnnoyIndex(embeds.shape[1], 'angular')  # Specify the distance metric
    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])
    search_index.build(10)  # Number of trees
    search_index.save(f'Index{method}.ann')  # Ensure 'data' directory exists
    return search_index



def search_article(query, search_index, df , Method):

    if Method == 'Mini':

          tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
          model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

          inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True, padding=True)
          with torch.no_grad():
              outputs = model(**inputs)
              hidden_states = outputs.last_hidden_state
          query_embed = hidden_states.mean(dim=1).squeeze().numpy()

          similar_item_ids, distances = search_index.get_nns_by_vector(query_embed, 20, include_distances=True)

          # Access similar articles
          search_results = [df.iloc[i]['pa_comment'] for i in similar_item_ids]
          #results_with_scores = list(zip(search_results, distances))
          return search_results , distances


Method = "Mini"

response = np.load('embeddingsMini.npy')
embeds = np.array(response)

Index = CreateIndex(embeds , Method) #create index

def ask_article_cohere(question, num_generations=1):

    # this is a function you call for generating the answer to the query.
    # You need a Cohere API code. which is free but with limmited calls

    co = cohere.Client("FZnpfTPjBLcPHietqOGFmLoQ4CiL8QCtSXbs6asu")

    context , _ = search_article(question, Index, df , Method)

    prompt = f"""
    Excerpt from the following article:
    {context}
    Question: {question}

    Extract the answer of the question from the text provided.
    If the text doesn't contain the answer,
    reply that the answer is not available."""

    prediction = co.generate(
        prompt=prompt,
        max_tokens=200,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )
    # return prediction.generations
    return [generation.text for generation in prediction.generations]






@app.route('/view_qa', methods=['GET'])
def view_qa():
    # Serve the HTML template with the chat UI
    return render_template('view_qa.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    print('Inside get_response()')
    data = request.get_json()
    query = data.get("queryInput", "")
    
    print('Received query input:')
    print(query)
    
    # Generate the response using your existing function
    response = ask_article_cohere(query, num_generations=1)
    if isinstance(response, list):
        answer = "\n".join(response)
    else:
        answer = response
    
    # Return the answer as JSON for the chat interface
    return jsonify({"answer": answer})



@app.route('/view_qa_old', methods=['GET', 'POST'])
def view_qa_old():
    print('Inside view_qa()')
    if request.method == 'POST':
        query = request.form.get('queryInput', "")  # Get the query input from the form
        print('Printing query input: ')
        print(query)
        #query = "tell me about peace agreements in Sudan"  # Get the query input from the form
        
        response  = ask_article_cohere(query, num_generations=1)
        #answer_text = ''.join(answer).replace('\n', '<br>')
        if isinstance(response, list):
            answer = "\n".join(response)  # Join the list elements into a string
        else:
            answer = response  # If it's already a string, use it as-is
        
        
        return render_template('view_qa.html', answer=answer)

    return render_template('view_qa.html')






@app.route('/', methods=['GET', 'POST'])
def index():
    selected_location = None
    selected_party_1 = None
    selected_party_2 = None
    selected_issue_level = "Level 1"
    conflicting_parties_1 = ["Select"]
    conflicting_parties_2 = ["Select"]

    if request.method == 'GET':
        selected_location = locations[0]
        conflicting_parties_1 = ["Select"]
        conflicting_parties_2 = ["Select"]
        count = 0
        with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            for row in reader: 
                count = count + 1; 
        
                if row[11] == '1000':
                    continue;
        
                if row[4] == selected_location:
                    if row[6] not in conflicting_parties_1:
                        conflicting_parties_1.append(row[6])
                        conflicting_parties_2.append(row[6])

        return render_template('index.html', locations=locations, 
                           conflicting_parties_1=conflicting_parties_1, 
                           conflicting_parties_2=conflicting_parties_2, 
                           selected_location=selected_location, 
                           selected_party_1=selected_party_1, 
                           selected_party_2=selected_party_2,
                           url='http://127.0.0.1:8080/view2.html',
                           qaurl='http://127.0.0.1:5000/view_qa',
                           issue_level=issue_level,
                           selected_issue_level=selected_issue_level)

    if request.method == 'POST':
        selected_location = request.form.get('location')
        selected_party_1 = request.form.get('selected_party_1')
        selected_party_2 = request.form.get('selected_party_2')
        selected_issue_level = request.form.get('selected_issue_level')
        print("Selected_location ", selected_location)
        print("Selected Party 1", selected_party_1)
        print("Selected Party 2", selected_party_2)
        print("Selected Issue Level", selected_issue_level)

        if selected_issue_level == "Level 1":
            column_index = 14
        if selected_issue_level == "Level 2":
            column_index = 15
        if selected_issue_level == "Level 3":
            column_index = 16

        conflicting_parties_1 = ["Select"]
        conflicting_parties_2 = ["Select"]
        count = 0
        with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            for row in reader: 
                count = count + 1; 
        
                if row[11] == '1000':
                    continue;
        
                if row[4] == selected_location:
                    if row[6] not in conflicting_parties_1:
                        conflicting_parties_1.append(row[6])
                        conflicting_parties_2.append(row[6])




        if selected_party_1 == "Select" and selected_party_2 == "Select":
            print('Rendering...')
            print(len(conflicting_parties_1))
            print(len(conflicting_parties_2))

            return render_template('index.html', locations=locations, 
                conflicting_parties_1=conflicting_parties_1, 
                conflicting_parties_2=conflicting_parties_2, 
                selected_location=selected_location, 
                selected_party_1=selected_party_1, 
                selected_party_2=selected_party_2,
                url='http://127.0.0.1:8080/view2.html',
                qaurl='http://127.0.0.1:5000/view_qa',
                issue_level=issue_level,
                selected_issue_level=selected_issue_level)


        l_selected_party2 = selected_party_2
        if l_selected_party2 == "Select":
            l_selected_party2 = selected_party_1 

        


        data_dict = {}
        with open(code_input, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                code, description = row
                data_dict[str(code)] = description.rstrip()


        color1 = '#248055'
        color2 = '#892433'
        color3 = '#552489'

        dyad1_id = []
        conflict1_id = []
        location1 = []
        side1_b = []
        year1 = []
        issue1_text = []

        dyad2_id = []
        conflict2_id = []
        location2 = []
        side2_b = []
        year2 = []
        issue2_text = []

        with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            for row in reader: 
                if row[11] == '1000':
                    #print(row[column_index]);
                    continue;

                if row[4] == selected_location and row[6] == selected_party_1:
                    dyad1_id.append(row[1])
                    conflict1_id.append(row[2])
                    location1.append(row[4])
                    side1_b.append(row[6])
                    year1.append(row[11])
                    issue1_text.append(data_dict[row[column_index]])


        with open(input_file, 'r', encoding='utf-8', errors='replace') as csvfile:
            reader = csv.reader(csvfile,delimiter=';')
            for row in reader: 
                if row[11] == '1000':
                    #print(row[column_index]);
                    continue;

                if row[4] == selected_location and row[6] == l_selected_party2:
                    dyad2_id.append(row[1])
                    conflict2_id.append(row[2])
                    location2.append(row[4])
                    side2_b.append(row[6])
                    year2.append(row[11])
                    issue2_text.append(data_dict[row[column_index]])


        output_file = 'templates/files/output.json'
        stored_side = []
        stored_year = []
        stored_side_issue = []
        stored_year_issue = []
        imported1_issues = []  

        with open(output_file, 'w', encoding='utf-8', errors='replace') as output:
                
            importStr = ''
            number_of_imports = 0
            last_year = 0
            output1Str = ''
            count = len(conflict1_id)-1
                
            for i in range(len(conflict1_id)):
                index = count - i    
                new_year = year1[index]
                if new_year not in stored_year:
                    stored_year.append(new_year)
                    if number_of_imports > 0:
                        #if index > 0:
                        output1Str = '{}{}"name":"conflict.{}({})","size":743,"color":"{}","imports":[{}]{}'.format(output1Str, '{', side1_b[index], last_year, color1, importStr, '},\n')
                        #else:
                        #    output1Str = '{}{}"name":"conflict.{}({})","size":743,"imports":[{}]{}'.format(output1Str, '{', dyad1_id[index], last_year, importStr, '}\n')
                            

                    last_year = new_year
                    number_of_imports = 1
                    importStr = '\"conflict.{}\"'.format(issue1_text[index])  
                    stored_year_issue = []
                    stored_year_issue.append(issue1_text[index])
                    if issue1_text[index] not in imported1_issues:
                        imported1_issues.append(issue1_text[index])
                            
                    continue
                        
                if index == 0:
                    number_of_imports = number_of_imports + 1 
                    if issue1_text[index] not in stored_year_issue:
                        importStr = '{}, \"conflict.{}\"'.format(importStr, issue1_text[index])  
                        stored_year_issue.append(issue1_text[index])
                        
                    if issue1_text[index] not in imported1_issues:
                        imported1_issues.append(issue1_text[index])
                            
                    output1Str = '{}{}"name":"conflict.{}({})","size":743,"color":"{}","imports":[{}]{}'.format(output1Str, '{', side1_b[index], last_year, color1, importStr, '},\n')
                            
                    break
                        
                number_of_imports = number_of_imports + 1 
                if issue1_text[index] not in stored_year_issue:
                    importStr = '{}, \"conflict.{}\"'.format(importStr, issue1_text[index])  
                    stored_year_issue.append(issue1_text[index])
                    
                if issue1_text[index] not in imported1_issues:
                    imported1_issues.append(issue1_text[index])
                
                
        # process dyad2 entries...
            stored_side = []
            stored_year = []
            stored_side_issue = []
            stored_year_issue = []
            imported2_issues = []  
            importStr = ''
            number_of_imports = 0
            last_year = 0
            #output1Str = ''
            count = len(conflict2_id)-1
                
            for i in range(len(conflict2_id)):
                index = count - i    
                new_year = year2[index]
                if new_year not in stored_year:
                    stored_year.append(new_year)
                    if number_of_imports > 0:
                        if index > 0:
                            output1Str = '{}{}"name":"conflict.{}({})","size":843,"color":"{}","imports":[{}]{}'.format(output1Str, '{', side2_b[index], last_year, color2, importStr, '},\n')
                        else:
                            output1Str = '{}{}"name":"conflict.{}({})","size":843,"color":"{}","imports":[{}]{}'.format(output1Str, '{', side2_b[index], last_year, color2, importStr, '}\n')
                            

                    last_year = new_year
                    number_of_imports = 1
                    importStr = '\"conflict.{}\"'.format(issue2_text[index])  
                    stored_year_issue = []
                    stored_year_issue.append(issue2_text[index])
                    if issue2_text[index] not in imported2_issues:
                        imported2_issues.append(issue2_text[index])
                            
                    continue
                        
                if index == 0:
                    number_of_imports = number_of_imports + 1 
                    if issue2_text[index] not in stored_year_issue:
                        importStr = '{}, \"conflict.{}\"'.format(importStr, issue2_text[index])  
                        stored_year_issue.append(issue2_text[index])
                        
                    if issue2_text[index] not in imported2_issues:
                        imported2_issues.append(issue2_text[index])
                            
                    output1Str = '{}{}"name":"conflict.{}({})","size":843,"color":"{}","imports":[{}]{}'.format(output1Str, '{', side2_b[index], last_year, color2, importStr, '}\n')
                            
                    break
                        
                number_of_imports = number_of_imports + 1 
                if issue2_text[index] not in stored_year_issue:
                    importStr = '{}, \"conflict.{}\"'.format(importStr, issue2_text[index])  
                    stored_year_issue.append(issue2_text[index])
                    
                if issue2_text[index] not in imported2_issues:
                    imported2_issues.append(issue2_text[index])

                
                
                
            totalCount = len(imported1_issues) + len(imported2_issues)
            output.write('[')
            count = len(imported1_issues) + 1
            for p_issues in imported1_issues:
                if p_issues not in imported2_issues:
                    totalCount = totalCount - 1 
                    output.write('{')
                    output.write('"name":"conflict.{}","size":{},"color":"{}","imports":[]'.format(p_issues, totalCount, color1))
                    output.write('},\n')
                        
            for p_issues in imported1_issues:
                if p_issues in imported2_issues:
                    totalCount = totalCount - 1 
                    output.write('{')
                    output.write('"name":"conflict.{}","size":{},"color":"{}","imports":[]'.format(p_issues, totalCount, color3))
                    output.write('},\n')

            for p_issues in imported2_issues:
                if p_issues not in imported1_issues:
                    totalCount = totalCount - 1 
                    output.write('{')
                    output.write('"name":"conflict.{}","size":{},"color":"{}","imports":[]'.format(p_issues, totalCount, color2))
                    output.write('},\n')

                        
            output.write(output1Str)
            output.write(']')
                                
        
        

        

    return render_template('index.html', locations=locations, 
                           conflicting_parties_1=conflicting_parties_1, 
                           conflicting_parties_2=conflicting_parties_2, 
                           selected_location=selected_location, 
                           selected_party_1=selected_party_1, 
                           selected_party_2=selected_party_2,
                           url='http://127.0.0.1:8080/view.html',
                           qaurl='http://127.0.0.1:5000/view_qa',
                           issue_level=issue_level,
                           selected_issue_level=selected_issue_level)

if __name__ == '__main__':
    app.run(debug=True)
