import os

filename = r"c:\Users\madhu\OneDrive\Desktop\SriCity\New folder\AyurSLM\app.py"

with open(filename, "r", encoding="utf-8") as f:
    content = f.read()

# Split the content at global_language = gr.Dropdown(
split_marker = '    global_language = gr.Dropdown('
parts = content.split(split_marker)

if len(parts) < 2:
    print("Error: Could not find split marker.")
    exit(1)

new_tail = """    global_language = gr.Dropdown(
        choices=["English", "Kannada (ಕನ್ನಡ)", "Hindi (हिंदी)", "Telugu (తెలుగు)", "Tamil (தமிழ்)", "Marathi (मराठी)"], 
        value="English", 
        label="🌐 Response Language / ಭಾಷೆ / भाषा / மொழி / భాష",
        interactive=True
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            master_desc_md = gr.Markdown("Fill out your complete profile once to instantly generate a comprehensive Ayurvedic Master Plan covering Dosha analysis, diet, herbs, daily routine, and yoga.", elem_classes=["prose"])
            master_symptoms = gr.Textbox(lines=2, label="Main Concern / Symptoms", placeholder="E.g., Chronic back pain, low energy, and stress")
            
            with gr.Row():
                master_age = gr.Number(label="Age", value=30)
                master_fitness = gr.Dropdown(choices=["Beginner", "Intermediate", "Advanced"], label="Fitness Level", value="Beginner")
            
            with gr.Row():
                master_loc = gr.Textbox(lines=1, label="Location / Region", placeholder="E.g., Delhi, Kerala, Texas")
                master_season = gr.Dropdown(choices=["Summer", "Monsoon", "Autumn", "Winter", "Spring"], label="Current Season", value="Summer")
            
            with gr.Row():
                master_diet = gr.Dropdown(choices=["Vegetarian", "Non-Vegetarian", "Vegan", "Mixed"], label="Diet", value="Vegetarian")
                master_digestion = gr.Dropdown(choices=["Irregular", "Strong/Acidic", "Slow/Heavy"], label="Digestion", value="Irregular")
            
            with gr.Row():
                master_sleep = gr.Dropdown(choices=["Light/Broken", "Sound/Short", "Deep/Heavy"], label="Sleep Pattern", value="Light/Broken")
                master_occ = gr.Textbox(lines=1, label="Occupation (e.g., Desk job)", placeholder="E.g., Software Engineer")
            
            master_btn = gr.Button("Generate Complete Master Plan", variant="primary")
            
    with gr.Tabs():
        with gr.TabItem("🩺 Symptom & Dosha Analyzer"):
            master_out_1 = gr.Textbox(lines=15, label="1. Dosha Analysis", interactive=False)
            
        with gr.TabItem("🌿 Herb & Remedy Finder"):
            master_out_2 = gr.Textbox(lines=15, label="2. Herbal Remedies", interactive=False)
            
        with gr.TabItem("🌦️ Seasonal Wellness (Ritucharya)"):
            master_out_3 = gr.Textbox(lines=15, label="3. Dietary Guidelines", interactive=False)
            
        with gr.TabItem("🌅 Dinacharya Planner"):
            master_out_4 = gr.Textbox(lines=15, label="4. Daily Routine (Dinacharya)", interactive=False)
            
        with gr.TabItem("🧘 Yoga & Pranayama Advisor"):
            master_out_5 = gr.Textbox(lines=15, label="5. Yoga & Breathwork", interactive=False)

        with gr.TabItem("🧘 Prakriti (Body Constitution) Test"):
            with gr.Row():
                with gr.Column(scale=1):
                    prakriti_desc_md = gr.Markdown("Select the options that best represent your *permanent, long-term nature*, not a temporary illness.")
                    body_type = gr.Dropdown(choices=["Thin/Slender, hard to gain weight", "Medium build, athletic", "Broad/Sturdy, gain weight easily"], label="Body type", allow_custom_value=True)
                    digestion_type = gr.Dropdown(choices=["Irregular, prone to gas/bloating", "Strong, intense hunger, prone to acidity", "Slow, steady, prone to heaviness"], label="Digestion pattern", allow_custom_value=True)
                    sleep_pattern = gr.Dropdown(choices=["Light, easily interrupted", "Sound but short, vividly dream", "Deep, heavy, hard to wake up"], label="Sleep pattern", allow_custom_value=True)
                    weather_pref = gr.Dropdown(choices=["Aversion to cold/wind", "Aversion to heat/sun", "Aversion to cold/damp"], label="Weather preference", allow_custom_value=True)
                    
                    prakriti_btn = gr.Button("Determine My Prakriti", variant="primary")
                with gr.Column(scale=1):
                    prakriti_output = gr.Textbox(lines=14, label="Your Prakriti & Lifestyle Tips", interactive=False)
                    
            prakriti_btn.click(generate_prakriti_analysis, inputs=[body_type, digestion_type, sleep_pattern, weather_pref, global_language], outputs=[prakriti_output])

        with gr.TabItem("📖 Context-Based Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    qa_desc_md = gr.Markdown("Paste an Ayurvedic text excerpt here and ask the AI a question about it. It will answer strictly based on your provided text.")
                    qa_context = gr.Textbox(lines=6, label="Ayurvedic Context", placeholder="Paste the reference text or book excerpt here...")
                    qa_question = gr.Textbox(lines=2, label="Your Question", placeholder="What does the text say about...")
                    qa_btn = gr.Button("Find Answer", variant="primary")
                with gr.Column(scale=1):
                    qa_output = gr.Textbox(lines=10, label="Answer", interactive=False)
                    
            qa_btn.click(generate_qa_response, inputs=[qa_context, qa_question, global_language], outputs=[qa_output])

        with gr.TabItem("💊 Drug Side Effects & Ayurveda"):
            with gr.Row():
                with gr.Column(scale=1):
                    drug_desc_md = gr.Markdown("Enter the name of a modern medicine to understand its side effects, its impact on your Doshas, and Ayurvedic ways to support your body while taking it.")
                    drug_name_in = gr.Textbox(lines=2, label="Medicine / Drug Name", placeholder="E.g., Paracetamol, Omeprazole, Ibuprofen")
                    drug_btn_in = gr.Button("Analyze Medicine", variant="primary")
                with gr.Column(scale=1):
                    drug_output_in = gr.Textbox(lines=14, label="Side Effects & Ayurvedic Analysis", interactive=False)
                    
            drug_btn_in.click(generate_drug_analysis, inputs=[drug_name_in, global_language], outputs=[drug_output_in])

    master_btn.click(
        generate_holistic_plan, 
        inputs=[master_symptoms, master_age, master_fitness, master_loc, master_season, master_diet, master_digestion, master_sleep, master_occ, global_language], 
        outputs=[master_out_1, master_out_2, master_out_3, master_out_4, master_out_5]
    )

    def update_ui_language(lang):
        t = ui_translations.get(lang, ui_translations["English"])
        return [
            gr.update(value=t.get("prakriti_desc")),
            gr.update(label=t.get("body_label")),
            gr.update(label=t.get("digestion_label")),
            gr.update(label=t.get("sleep_label")),
            gr.update(label=t.get("weather_label")),
            gr.update(value=t.get("prakriti_btn")),
            gr.update(label=t.get("prakriti_out")),
            
            gr.update(value=t.get("qa_desc")),
            gr.update(label=t.get("qa_context_lbl"), placeholder=t.get("qa_context_pl")),
            gr.update(label=t.get("qa_question_lbl"), placeholder=t.get("qa_question_pl")),
            gr.update(value=t.get("qa_btn")),
            gr.update(label=t.get("qa_out")),
            
            gr.update(value=t.get("drug_desc")),
            gr.update(label=t.get("drug_name_lbl")),
            gr.update(value=t.get("drug_btn")),
            gr.update(label=t.get("drug_out")),
            
            gr.update(value=t.get("master_desc")),
            gr.update(label=t.get("master_sym_lbl")),
            gr.update(value=t.get("master_btn")),
            gr.update(label=t.get("m_out_1", "1. Dosha Analysis")),
            gr.update(label=t.get("m_out_2", "2. Herbal Remedies")),
            gr.update(label=t.get("m_out_3", "3. Dietary Guidelines")),
            gr.update(label=t.get("m_out_4", "4. Daily Routine (Dinacharya)")),
            gr.update(label=t.get("m_out_5", "5. Yoga & Breathwork"))
        ]
        
    global_language.change(
        update_ui_language,
        inputs=[global_language],
        outputs=[
            prakriti_desc_md, body_type, digestion_type, sleep_pattern, weather_pref, prakriti_btn, prakriti_output,
            qa_desc_md, qa_context, qa_question, qa_btn, qa_output,
            drug_desc_md, drug_name_in, drug_btn_in, drug_output_in,
            master_desc_md, master_symptoms, master_btn, master_out_1, master_out_2, master_out_3, master_out_4, master_out_5
        ]
    )

if __name__ == "__main__":
    print("Launching AyurSLM Interface...")
    demo.launch()
"""

new_content = parts[0] + new_tail

with open(filename, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Successfully replaced formatting in app.py")
