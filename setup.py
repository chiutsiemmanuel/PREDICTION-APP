from cx_Freeze import setup, Executable

# Define the main script and its dependencies
build_options = {
    "packages": ["pandas", "numpy", "PyQt5", "statsmodels", "mplcursors"],
    "excludes": [],
    "include_files": []  # Add any additional files (e.g., data files, icons) here
}

# Define the executable
executables = [
    Executable(
        script="app.py",  # Your main script
        base="Win32GUI",  # Use "Win32GUI" for no console window (GUI apps)
        icon="app_icon.ico"  # Optional: Add an icon file
    )
]

# Run the setup
setup(
    name="DataApp",
    version="1.0",
    description="Data Visualization & Prediction App",
    options={"build_exe": build_options},
    executables=executables
)
