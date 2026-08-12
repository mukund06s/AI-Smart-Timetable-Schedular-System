import pypdf
import os

pdf_path = r"D:\sts\College\Test output\New\AIDS\timetable_Sem1__Sec_B (1).pdf"
pdf_path_a = r"D:\sts\College\Test output\New\AIDS\timetable_Sem1__Sec_A (1).pdf"

for path in [pdf_path_a, pdf_path]:
    print(f"=== {os.path.basename(path)} ===")
    reader = pypdf.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Just print the first 1000 characters to get a sense of the layout,
    # and then count specific subjects
    print("TEXT SNIPPET:")
    print(text[:500])
    
    print("\nCOUNTS:")
    print(f"Calculus: {text.count('Calculus')}")
    print(f"Physics: {text.count('Physics')}")
    print(f"BEE: {text.count('Basic Electrical')}")
    print(f"Comp Thinking: {text.count('Computational Thinking')}")
    print(f"Eng Graphics: {text.count('Engineering Graphics')}")
    print(f"CC -201: {text.count('CC -201')}")
    print(f"MJ: {text.count('MJ')}")
    print("-" * 40)
