import os
from typing import List
from pypdf import PdfReader
#from langchain_community.document_loaders import PyPDFium2Loader
from logging_config import setup_logger 
logger = setup_logger(pkgname="rag_database")


class DataCollect:
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.folder = os.path.join(os.getcwd(),self.folder_name)
        self.total_files = self.check_file_names()
        self.file_checker = FolderChecker(folder_path= self.folder, file_collection = self.total_files)
        _, self.files = self.file_checker._check_unique_files()
        

    def check_file_names(self) -> List[str] :
        logger.info(f"Checking files in {self.folder_name} folder")
        files = []
        for file in os.listdir(self.folder):
            if file.endswith(('.pdf','.txt')):
                files.append(file)
        logger.info(f"{len(files)} files in {self.folder_name} folder")
        #self.check_new_files()
        return files
                             
    def check_new_files(self):
        logger.info(f"Checking any new files inside {self.folder_name}")
        self.unique_file_count, self.files = self.file_checker.check_unique_files()

    def load_content(self, save_text: bool = True) -> str:
        text_content = ""
        logger.info(f"Loading content from {self.folder_name} folder")
        for file in self.files:
            file_path = os.path.join(self.folder,file)
            #logger.info(f"filepath: {file_path}")
            try:
                if file_path.endswith('.pdf'):
                    text = self.load_pdf(file_path)
                    text_content+= text
                elif file_path.endswith('.txt'):
                    text = self.load_txt(file_path)
                    text_content+= text
                if save_text:
                    self.save_text(text_content=text_content)
                logger.info(f"Content loaded successfully")
            except Exception as e:
                raise Exception(f"Error processing {file_path}: {e}")
        if not text_content:
            logger.error("[Error] No text content loaded from files. Ensure the files contain valid text.")
            #raise ValueError("No text content loaded from files. Ensure the files contain valid text.")
        return text_content
            

    def load_pdf(self, file_path):
        logger.info(f"Extracting content from {os.path.basename(file_path)}")
        try:
            if not os.path.exists(file_path):
                logger.error(f"{file_path} not found")
                raise FileNotFoundError(f"{file_path} not found")
            reader = PdfReader(file_path)
            #reader = PyPDFium2Loader(file_path)
            text = ""
            #for page in reader.load():
            #    print("langchain", page)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                #text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            raise Exception(f"Error processing PDF file {file_path}: {e}")

    def load_txt(self, file_path):
        logger.info(f"Extracting content from {os.path.basename(file_path)}")
        try:
            if not os.path.exists(file_path):
                logger.error(f"{file_path} not found")
                raise FileNotFoundError(f"{file_path} not found")
            text = ""
            with open(file_path, 'r') as f:
                text += f.read()
            return text
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            raise Exception(f"Error processing text file {file_path}: {e}")

    def save_text(self, text_content: str, text_file: str = "extracted_text.txt"):
        logger.info(f"Saving extracted content to {text_file}")
        with open(text_file,"w") as f:
            f.write(text_content)
        logger.info(f"Saved extracted content to {text_file}")

    def get_text(self, save_text: bool) -> str:
        """
        Args:
            save_text (bool): save text flag
        Returns:
            str: Extracted text from files
        """
        logger.info(f"Getting extracted content")
        extracted_text = self.load_content(save_text=save_text)
        logger.info(f"Extracted content from files successfully")
        return extracted_text 
                        

        
class FolderChecker:
    def __init__(self, folder_path: str, file_collection: List[str], file_list_file: str = "file_list.txt"):
        self.folder_path = folder_path
        self.file_collection = file_collection
        self.file_list_file = file_list_file
        self.file_list_file_fullpath = os.path.join(os.path.split(folder_path)[0], self.file_list_file)
         # Create the file if it doesn't exist
        try:
            if not os.path.exists(self.file_list_file_fullpath):
                logger.info(f"Creating {file_list_file}")
                with open(self.file_list_file_fullpath, "w") as f:
                    f.write("")
                logger.info(f"Created {file_list_file} successfully")
        except FileNotFoundError as e:
            raise Exception(f"Unable to find {self.file_list_file}: {e}")  
        #self.time_stamps = self.create_file_for_timestamps()

    def create_file_for_timestamps(self, timestamp_file:str = "time_stamp.txt"):
        self.timestamp_file_path = os.path.join(os.path.split(self.file_path)[0], timestamp_file)
        if not os.path.exists(self.timestamp_file_path):
            with open(self.timestamp_file_path, "w") as f:
                f.write("") 
    
    def _check_unique_files(self):
        try:
            logger.info(f"Checking for uniques files in {self.folder_path}")
            with open(self.file_list_file, 'r') as f:
                existing_files = set(f.read().splitlines())
        except IOError as e:
            raise Exception(f"[Error]: An unexpected I/O error occurred while reading {self.file_list_file}: {e}")
            #existing_files = set()
        unique_files = set(self.file_collection) - existing_files
        if unique_files:
            logger.info(f"Appending {len(unique_files)} unique files to {self.file_list_file}")
            with open(self.file_list_file, 'a') as f:
                f.writelines(f"{file}\n" for file in unique_files)
                logger.info(f"Successfully appended {len(unique_files)} unique files to {self.file_list_file}")
            #with open(self.timestamp_file_path, 'a') as f:
            #    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #    f.write(f"{current_time}\n")
        else:
            if len(list(unique_files)) > 0:
                logger.info(f"List of new files: {list(unique_files)}")
            else:
                logger.info("No new files found.")
        return len(unique_files), unique_files




if __name__=="__main__":
    folder = "new_files"

    DataCollect(folder=folder)
