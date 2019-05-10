import sys 
import os

class FACLogger():

    logger_path = None
    log = None
    terminal = None

    def __init__(self, path:str, log_name:str):
        try:
            self.terminal = sys.stdout
            self.logger_path:str = path + log_name + ".log"

            print("Path:", self.logger_path)
            
            if not os.path.exists(self.logger_path):
                open(self.logger_path, "x").close()                    

            if os.path.exists(self.logger_path):
                self.log = open(self.logger_path, "a")
                print("Logger SET:", self.logger_path)
            else:
                print("Logger NOT:", self.logger_path)
        except Exception as ex:
            template = "An exception of type {0} occurred in [Logger.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def write(self, context:str):
        try:
            self.terminal.write(context)
            self.log.write(context)  
        except Exception as ex:
            template = "An exception of type {0} occurred in [Logger.Write]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def flush(self):
        pass    

class FolderCreator():

    def __init__(self, path:str):
        try:
            self.folder_path:str = path
        except Exception as ex:
            template = "An exception of type {0} occurred in [FolderCreator.Constructor]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

    def Create(self):
        try:
            if not os.path.exists(self.folder_path) or not os.path.isdir(self.folder_path):
                os.mkdir(self.folder_path)
                print("Folder ["+ self.folder_path +"] created!")
            else:
                print("Folder ["+ self.folder_path +"] already exists!")

            return os.path.isdir(self.folder_path)
        except Exception as ex:
            template = "An exception of type {0} occurred in [FolderCreator.Create]. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return False