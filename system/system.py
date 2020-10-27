import sys
import os
import wikipedia
import shutil
import time
import pandas as pd
from haystack import Finder
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.retriever.sparse import TfidfRetriever
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers

def fetch_documents(book_title):
    # Function to fetch relevant documents given a book title

    if os.path.isdir("documents"):
        shutil.rmtree('documents')
    os.mkdir('documents')

    page_counter = 1

    for counter, title in enumerate(wikipedia.search(book_title, results=50)):
        exclusions = ("film", "video game", "album", "soundtrack")

        if not any(x in title for x in exclusions):
            try:
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content
                path = os.path.join('documents', str(page_counter)+'.txt')
                f = open(path, 'w', encoding='utf-8')
                f.write(content)
                f.close()
                print('Created document number ' + str(page_counter)
                    + ' from page ' + title)
                page_counter += 1
            except:
                pass

    return page_counter


if __name__ == "__main__":
    # Load books and questions csv files
    book_df = pd.read_csv('books.csv', encoding='utf-8')
    q_df = pd.read_csv('questions.csv', encoding='utf-8')

    # Get document ids
    doc_ids = book_df['document_id'].values

    # Create DataFrame for books
    doc_df = pd.DataFrame(columns = ['document_id', 'wiki_title', 'num_of_docs',
        'set_up_time'])

    # Create DataFrame for question-answers
    qa_df = pd.DataFrame(columns = ['document_id', 'question_id', 'question',
        'answer', 'probability', 'score', 'top_k_retriever', 'time_taken'])

    # Use transfromer reader
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
        use_gpu=True)

    # Loop over the books
    for doc_id in doc_ids:
        begin = time.time()

        # Fetch documents based on book title
        book_title = book_df[book_df['document_id'] == doc_id
            ]['wiki_title'].iloc[0]
        print('Fetching documents for book ' + book_title)
        num_docs = fetch_documents(book_title)

        # Create an in-memory document store and add the files
        document_store = InMemoryDocumentStore()
        doc_dir = "documents"
        dicts = convert_files_to_dicts(dir_path=doc_dir,
            clean_func=clean_wiki_text, split_paragraphs=True)

        document_store.write_documents(dicts)

        # Use TF-IDF retriever
        retriever = TfidfRetriever(document_store=document_store)

        finder = Finder(reader, retriever)

        end = time.time()

        # Add a row to the document DataFrame
        doc_df = doc_df.append({'document_id': doc_id, 'wiki_title': book_title,
            'num_of_docs': num_docs, 'set_up_time': end-begin},
            ignore_index = True)

        # Get the questions for the given document
        bq_df = q_df[q_df['document_id'] == doc_id]
        questions = bq_df['question'].values

        for question in questions:
            begin = time.time()
            top_k_retriever = 7
            prediction = finder.get_answers(question=question,
                top_k_retriever=top_k_retriever, top_k_reader=1)
            j = prediction
            end = time.time()
            qa_df = qa_df.append({'document_id': doc_id,
                'question_id': bq_df[bq_df['question'] == question].index[0],
                'question': question, 'answer': j['answers'][0]['answer'],
                'probability': j['answers'][0]['probability'],
                'score': j['answers'][0]['score'],
                'time_taken': end-begin},
                ignore_index = True)

    # Save the dataframes in a folder, saying specifications
    path_name = os.path.join('in-memory-document-store', 'roberta-base-squad-2',
        'top-7-retriever', '50-wiki-results')
    os.makedirs(path_name)

    doc_df.to_csv(os.path.join(path_name, 'books.csv'), encoding='utf-8')
    qa_df.to_csv(os.path.join(path_name, 'questions.csv'), encoding='utf-8')
