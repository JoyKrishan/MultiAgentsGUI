import os 

def list_java_files(main_dir) -> list:
    directory = main_dir
    java_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".java"):
                java_files.append(os.path.join(root.replace("{}/".format(main_dir), ""), file))

    return java_files


def preprocess_paths(project_name, bug_index, filepath):
    project_dir = os.path.join("workspace", project_name.lower()+"_"+str(bug_index)+"_buggy")
    
    if filepath.endswith(".java"):
        filepath = filepath[:-5]
        filepath = filepath.replace(".", "/")
        filepath += ".java"
    else:
        filepath = filepath.replace(".", "/")
        
        if not os.path.exists(os.path.join(project_dir,filepath)):  # If the filepath cannot be found, we will search for it in the files_index.txt
            if not os.path.exists(os.path.join(project_dir, "files_index.txt")):
                with open(os.path.join(project_dir, "files_index.txt"), "w") as fit:
                    fit.write("\n".join(list_java_files(project_dir)))
                
            with open(os.path.join(project_dir, "files_index.txt")) as fit:
                files_index = [f for f in fit.read().splitlines() if filepath in f]
            
            if len(files_index) == 1:
                filepath = files_index[0]
            elif len(files_index) >= 1:
                raise ValueError("Multiple Candidate Paths. We do not handle this yet!")
            else:
                return "The filepath {} does not exist.".format(filepath)
    return filepath

print(preprocess_paths("Closure", "10", "src.com.google/javascript/jscomp/CommandLineRunner.java"))