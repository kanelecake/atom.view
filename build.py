import PyInstaller.__main__
import shutil

PyInstaller.__main__.run([
    'main.py',
    '--onefile',
    '--noconsole',
], )

shutil.copytree('required/', 'dist/required')
shutil.copytree('models/', 'dist/models')
