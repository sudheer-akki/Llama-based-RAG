from data_collection import DataCollect
import re
from nltk.tokenize import word_tokenize
from logging_config import setup_logger 
from typing import List
logger = setup_logger(pkgname="rag_database")



class TextProcess():
    def __init__(self,
        folder_name: str = "new_files",
        text_file: str = "extracted_text.txt",
        stopwords_file: str = "stop_words.txt",
        ):
        self.cleaned_text = "cleaned_text.txt"
        self.chunk_text_file = "chunked.txt"
        self.stopwords_file = stopwords_file
        self.data_collect = DataCollect(folder_name=folder_name)
        self.extracted_text = self.data_collect.get_text(text_file=text_file, save_text=True)
        
    def _clean_and_chunk_content(self, 
            chunk_size: int = 50,
            overlap_size: int = 10,
            save: bool = True) -> List[str]:
        try:
            if not isinstance(self.extracted_text, str):
                logger.error(f"Extracted text is not in string format")
                raise ValueError("The content returned must be a string.")
            # Clean each line: remove leading numbers and strip whitespace
            logger.info(f"Cleaning extracted text")
            cleaned_lines = [
                re.sub(r'^\d+\.\s*', '', line).strip()
                for line in self.extracted_text.splitlines() if line.strip()
            ]
            # Join cleaned lines back into a single string
            cleaned_text = '\n'.join(cleaned_lines)
            cleaned_text = self._clean_text(content = cleaned_text)
            if save:
                with open(self.cleaned_text,"w") as f:
                    f.write(cleaned_text)
            chunked_text = self._chunk_text(
                cleaned_text=cleaned_text,
                chunk_length=chunk_size,
                overlap_count=overlap_size,
                save_chunk = True
                )
            return chunked_text
        except Exception as e:
            print(f"An error occurred: {e}")

    def _clean_text(self,content) -> str:
        #text_ = self._remove_stopwords(
        #    text = content, 
        #    stopword_file = self.stopwords_file
        #    )
        logger.info(f"Further cleaning started")    
        # Remove email addresses (handle special characters)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', '', content)
        # Remove distracting single quotes
        text = re.sub("\'", '', text)
        # Remove references like [30], [1], etc
        text = re.sub(r'\[\d+\]', '', text)
        # Remove enumerations like (i), (ii), (iii), (1), (a), (b), etc.
        text = re.sub(r'\([A-Za-z0-9]+\)', '', text)
        # Remove citations like (Smith et al., 2020)
        text = re.sub(r'\([A-Za-z]+ et al\., \d{4}\)', '', text)
        # Remove 'Table' and 'Figure' references like 'As shown in Table 1'
        text = re.sub(r'\bTable \d+\b', '', text)
        text = re.sub(r'\bFigure \d+\b', '', text)
        # Remove footnotes like * or †, plus any additional special symbols (‡, §, etc.)
        text = re.sub(r'[\*\†‡§]', '', text)
        # Remove URLs or DOIs
        text = re.sub(r'http[s]?://\S+|doi:\S+', '', text) 
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove unnecessary punctuations but preserve hyphens in compound words and numerical ranges
        text = re.sub(r'[^\w\s.,;:-]', '', text)  # Keep .,;:- for punctuation in text
        # Remove unnecessary punctuations and extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        # Replace hyphens with no space, unless part of a compound word (e.g., high-quality)
        text = re.sub(r'(?<!\w)-|-(?!\w)', '', text)
        # Remove patterns like [14, 18] including numbers
        text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
        #text = re.sub(r'\[\d+,\s*\d+\]', '', text)
        # Remove multiple spaces, newlines, or special characters if needed
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
        text = text.lower()
        logger.info(f"Further cleaning completed")       
        return text.strip()
    
    def _remove_stopwords(self, text, stopword_file: str) -> str:
        logger.info(f"Removing stopwords")
        try:
            with open(stopword_file, "r") as f:
                stop_words = f.read()
        except IOError as e:
            logger.error(f"Unable to open {file}: {e}")
            raise Exception(f"Unable to open {file}: {e}")
        stop_words = stop_words.replace('\n', ' ').split(' ')
        logger.info(f"Tokenizing text")
        # Tokenize the input text
        words = word_tokenize(text)
        # Remove stopwords
        filtered_words = [word for word in words if word.lower() not in stop_words]
        logger.info(f"Removed stopwords successfully")
        # Join words back into a single string
        return ' '.join(filtered_words)


    def _chunk_text(self, 
            cleaned_text: str, 
            chunk_length: int = 50,
            overlap_count: int = 10,
            save_chunk=True) -> List[str]:
        logger.info(f"Chunking extracted text")
        words = cleaned_text.split()
        # Initialize the list to store chunks
        chunks = []
        # Iterate through the text with a step size that considers overlap
        i = 0
        while i < len(words):
            # Take the next chunk based on max_length, including overlap
            chunk = words[i:i + chunk_length]
            chunks.append(' '.join(chunk))
            # Move the starting index for the next chunk by max_length - overlap_count
            i += chunk_length - overlap_count
        if save_chunk:
            # Save the chunks to a file line by line
            with open(self.chunk_text_file, 'w') as file:
                for chunk in chunks:
                    file.write(chunk.strip() + '\n')
        return chunks


if __name__=="__main__":

    cleaned_te = TextProcess(chunk_length=10, overlap_length= 3)

    chunks = cleaned_te.chunk_text()
    print("length", len(chunks))

    with open("chunked.txt", 'w') as file:
        for chunk in chunks:
            file.write(chunk.strip() + '\n')
