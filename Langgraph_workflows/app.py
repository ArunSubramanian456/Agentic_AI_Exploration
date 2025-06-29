from src.langgraphagenticai.main import load_langgraph_agenticai_app
import os
import streamlit as st

def check_file_loading():
    st.write("---")
    st.subheader("File System Debug Info")

    # 1. Print the current working directory
    current_working_dir = os.path.dirname(os.path.abspath(__file__))  # <-- **Replace with your actual path**)
    st.write(f"**Current Working Directory:** `{current_working_dir}`")

    # 2. List all files and directories in the current working directory
    try:
        files_and_dirs = os.listdir(current_working_dir)
        st.write("**Files and directories in current directory:**")
        st.code('\n'.join(sorted(files_and_dirs)))
    except Exception as e:
        st.error(f"Error listing files: {e}")

    # 3. Check for the existence of your config file
    config_file_path = "./src/langgraphagenticai/ui/uiconfigfile.ini"  # <-- **Replace with your actual path**
    if os.path.exists(config_file_path):
        st.success(f"**Config file found at:** `{config_file_path}`")
        # You can even try to read it to confirm it's readable
        try:
            with open(config_file_path, "r") as f:
                content = f.read()
                st.write("**Config file content:**")
                st.code(content)
        except Exception as e:
            st.warning(f"Found config file, but could not read it: {e}")
    else:
        st.error(f"**Config file NOT found at:** `{config_file_path}`")


if __name__=="__main__":
    check_file_loading()
    # load_langgraph_agenticai_app()