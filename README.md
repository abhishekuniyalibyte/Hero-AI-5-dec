create env: python3.10 -m venv myenv
act env: source myenv/bin/activate

menu_extraction.py: extract menu from pdf/jpeg file
menu_embeddding_generator.py: generate embedding from file created by menu_extraction.py
chatbot.py: chatbot that will read embedding file to answer user question.

run file: python3 menu_extraction.py menu2.jpg

TASK TO BE DONE:
-> show MENU as list
-> make it iteration based (for eg: add on gulab jamun in thali)
-> with pizaa add toppins
-> calories (based on average)
-> in meals what do you want (roti, which sabzi)