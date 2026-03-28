import gradio as gr
import torch
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import pipeline

max_seq_length = 2048
dtype = None # Auto detection
load_in_4bit = True

model_name = "unsloth/Llama-3.2-1B-Instruct" 
lora_dir = "ayurslm-lora"

print("Loading AyurSLM...")

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    if os.path.exists(lora_dir):
        print("Found fine-tuned LoRA adapters, loading...")
        model.load_adapter(lora_dir)
    else:
        print(f"Warning: Did not find fine-tuned adapters at {lora_dir}.")
        print("Loading base model only. Consider running train.py first.")
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
else:
    print("No GPU detected. Falling back to standard CPU inference using Transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    if os.path.exists(lora_dir):
        print("Found fine-tuned LoRA adapters, loading CPU version...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)

print("Loading Offline Resources Database and Embedding Models...")
try:
    chroma_client = chromadb.PersistentClient(path="./ayur_offline_db")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 
    offline_collection = chroma_client.get_or_create_collection(name="ayurveda_offline_docs")
    interaction_cache = chroma_client.get_or_create_collection(name="user_interaction_cache")
    
    # Auto-load files from offline_resources
    offline_dir = "offline_resources"
    if os.path.exists(offline_dir):
        files = glob.glob(os.path.join(offline_dir, "*"))
        # Improved: Check if we need to update/re-index (simple check: if file count changed or collection is empty)
        # For a more robust solution, we'd track file hashes, but for now, we'll allow forcing a re-index if a file named 'FORCE_REINDEX' exists
        force_reindex = os.path.exists(os.path.join(offline_dir, "FORCE_REINDEX"))
        
        if (offline_collection.count() == 0 and files) or force_reindex:
            if force_reindex:
                print("Force re-index requested. Clearing old database...")
                chroma_client.delete_collection("ayurveda_offline_docs")
                offline_collection = chroma_client.create_collection(name="ayurveda_offline_docs")
                os.remove(os.path.join(offline_dir, "FORCE_REINDEX"))

            print("Extracting and vectorizing offline resources...")
            docs = []
            ids = []
            for file_path in files:
                if os.path.basename(file_path) == "FORCE_REINDEX": continue
                try:
                    text = ""
                    if file_path.endswith('.pdf'):
                        reader = PdfReader(file_path)
                        for page in reader.pages:
                            if page.extract_text():
                                text += page.extract_text() + "\n"
                    else:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            
                    words = text.split()
                    # Smaller chunks for better matching (100 words)
                    chunks = [' '.join(words[i:i + 100]) for i in range(0, len(words), 100)]
                    doc_idx = os.path.basename(file_path)
                    for chunk_id, chunk_text in enumerate(chunks):
                        if len(chunk_text.strip()) > 20: # Skip very small chunks
                            docs.append(chunk_text)
                            ids.append(f"{doc_idx}_{chunk_id}_{hash(chunk_text)}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
            
            if docs:
                # Add in batches to avoid overhead
                batch_size = 100
                for i in range(0, len(docs), batch_size):
                    batch_docs = docs[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]
                    offline_collection.add(
                        documents=batch_docs, 
                        embeddings=embedding_model.encode(batch_docs).tolist(), 
                        ids=batch_ids
                    )
                print(f"Successfully indexed {len(docs)} chunks from {len(files)} files.")
        else:
            print(f"Offline database loaded with {offline_collection.count()} knowledge chunks.")
except Exception as e:
    print(f"Offline RAG init failed: {e}")

def get_offline_context(query, n_results=3):
    """Helper to fetch relevant knowledge chunks from the offline database."""
    try:
        if offline_collection.count() == 0:
            return "No offline knowledge available. Please check the 'offline_resources' folder."
            
        results = offline_collection.query(
            query_embeddings=[embedding_model.encode(query).tolist()],
            n_results=n_results
        )
        if results['documents'] and results['documents'][0]:
            context = "\n---\n".join(results['documents'][0])
            print(f"RAG: Found {len(results['documents'][0])} matches for query: '{query[:50]}...'")
            return context
    except Exception as e:
        print(f"Context retrieval error: {e}")
    return "No specific matching text found in traditional resources."

def check_cache(query_key):
    """Check if an identical or near-identical query exists in the cache."""
    try:
        if interaction_cache.count() == 0:
            return None
        res = interaction_cache.query(
            query_embeddings=[embedding_model.encode(query_key).tolist()],
            n_results=1
        )
        # 0.98 similarity threshold for 'identical'
        if res['distances'] and res['distances'][0] and res['distances'][0][0] < 0.02:
            print(f"Cache Hit! Serving stored response for: {query_key[:50]}...")
            return res['metadatas'][0][0]
    except Exception as e:
        print(f"Cache check error: {e}")
    return None

def save_to_cache(query_key, response_data, context_data):
    """Save a successful generation to the interaction cache."""
    try:
        import json
        payload = json.dumps({"response": response_data, "context": context_data})
        interaction_cache.add(
            documents=[query_key],
            embeddings=[embedding_model.encode(query_key).tolist()],
            metadatas=[{"data": payload}],
            ids=[f"cache_{hash(query_key)}_{os.urandom(4).hex()}"]
        )
    except Exception as e:
        print(f"Cache save error: {e}")

def apply_hallucination_filters(text):
    if not isinstance(text, str):
        return text
    filters = [
        "As an AI language model",
        "I cannot provide medical advice",
        "consult a doctor",
        "consult your physician",
        "western medicine",
        "FDA approved",
        "allopathic",
        "modern medicine"
    ]
    import re
    for f in filters:
        text = re.sub(f"(?i){re.escape(f)}", "", text)
    return text.strip()

ayur_prompt = """Namaste. You are a conversational, friendly, and expert Ayurvedic assistant. Below is an instruction detailing a patient's symptoms or questions. 

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Patient Context:
- Location (in India): {location}
- Food Habitat: {diet}
- Current Season: {season}

*** CRITICAL SAFETY CHECK ***
First, analyze the user input for serious or emergency symptoms.
If the symptoms include ANY of the following:
- Chest pain
- Severe breathing issues
- Unconsciousness
- Heavy bleeding

Then:
1. Do NOT give home remedies or Ayurvedic suggestions.
2. Clearly advise immediate medical attention and to visit a hospital or call emergency services.
Stop all other analysis.

Otherwise, proceed safely with the Ayurvedic task:
1. Express deep EMPATHY and COMPASSION. Start by warmly validating what they are going through, making them feel heard, comforted, and cared for.
2. Identify the possible Dosha imbalance (Vata, Pitta, or Kapha) based on the symptoms. Explain briefly WHY this imbalance is likely.
3. Suggest TRADITIONAL AYURVEDIC remedies heavily CUSTOMIZED to their Food Habitat ({diet}), Geographical Location ({location}), and Current Season ({season}):
   - Ayurvedic Diet: Recommend foods and diet changes strictly suited to a {diet} diet, taking into account the local cuisine and current seasonal weather in {location}.
   - Ayurvedic Lifestyle: Suggest specific daily routines adapted to the current climate of {location}.
   - Traditional Home Remedies: Recommend remedies using only herbs, spices, and ingredients that are locally and naturally available in {location}.

IMPORTANT: For EVERY suggestion you give, you MUST include:
- Reason: Why this remedy is recommended in Ayurveda (be specific and educational, avoid generic modern answers).
- Effect: How it helps balance the specific Dosha.

STRICT RULE: Do NOT suggest modern medical solutions, western medicine, or generic modern lifestyle advice (e.g., "drink more water", "take painkillers", "use a humidifier"). Provide strictly Ayurvedic-based suggestions. Maintain a deeply compassionate and friendly tone.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use very short, simple sentences.
- Avoid technical terms or complex Sanskrit without explaining them simply.
- Use everyday examples to explain concepts.
- Make it extremely easy to understand for beginners or non-educated users.

### User input:
{input}

### Response:
"""

def generate_ayurvedic_advice(instruction, location, diet, season, language):
    import json
    query_key = f"ADVICE|{instruction}|{location}|{diet}|{season}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response'], data['context']

    context = get_offline_context(instruction)
    inputs = tokenizer(
        [ayur_prompt.format(language=language, input=instruction, location=location, diet=diet, season=season, retrieved_context=context)], return_tensors = "pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(
        **inputs, 
        max_new_tokens = 450, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

def process_interaction(text_input, location, diet, season, language):
    user_input = text_input.strip() if text_input else ""
                
    if not user_input:
        return text_input, "Please provide your symptoms via text.", ""
        
    res, ctx = generate_ayurvedic_advice(user_input, location, diet, season, language)
    return user_input, res, ctx

prakriti_prompt = """Namaste. You are a conversational, friendly, and expert Ayurvedic assistant. Below are details about an individual's body and habits.
Determine their Prakriti (body constitution).

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Data:
- Body type: {body}
- Digestion: {digestion}
- Sleep pattern: {sleep}
- Weather preference: {weather}

Your task is to:
1. Classify them as Vata, Pitta, or Kapha dominant (or a combination).
2. Explain your reasoning.
3. Suggest personalized TRADITIONAL AYURVEDIC lifestyle tips and herbs suitable for their dominant Dosha. 

STRICT RULE: Do NOT provide modern generic advice. Focus exclusively on Ayurvedic practices, diet laws, and traditional habits. Provide a concise, accurate, and structured response. Maintain a friendly tone.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use very short, simple sentences.
- Avoid technical terms or complex Sanskrit without explaining them simply.
- Use everyday examples to explain concepts.
- Make it extremely easy to understand for beginners or non-educated users.

### Response:
"""

def generate_prakriti_analysis(body, digestion, sleep, weather, language):
    import json
    query_key = f"PRAKRITI|{body}|{digestion}|{sleep}|{weather}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response']

    if not body or not digestion or not sleep or not weather:
        return "Please fill out all 4 fields to assess your Prakriti."
    
    context = get_offline_context(f"Body: {body}, Digestion: {digestion}, Sleep: {sleep}")
    instruction = prakriti_prompt.format(body=body, digestion=digestion, sleep=sleep, weather=weather, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 400, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

qa_prompt = """Namaste. You are a highly professional and expert Offline Ayurvedic AI Agent.
Analyze the following retrieved context and answer the user's question with utmost clarity and structure.

### Relevant Context:
{context}

### User Question:
{question}

Your task is to structure your response EXACTLY like this:

### 💡 Core Answer
[Provide a direct, completely clear, and concise 2-3 sentence answer to the user's question based on the text.]

### 🔍 Key Details
- [Bullet point 1 extracted from the text]
- [Bullet point 2]
- [Bullet point 3]

### 🌿 Traditional Context
[Briefly explain the underlying Ayurvedic principle (like Dosha or Agni) related to this topic, if mentioned in the context.]

STRICT RULES:
1. Base your answer ONLY on the context. If the answer is not in the text, respond: "I cannot find this information in the offline Ayurvedic database."
2. Do not use complex jargon. Keep it exceptionally easy to read.
3. LANGUAGE RULE: Output entirely in {language}.

### Response:
"""

def generate_qa_response(context, question, language):
    if not context.strip() or not question.strip():
        return "Please provide both the context and your question.", ""
        
    instruction = qa_prompt.format(context=context, question=question, language=language)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 300, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    return final_res, context

def ask_offline_agent(question, language):
    """Searches the offline database for the answer and formats it."""
    import json
    query_key = f"QA|{question}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response']

    if not question.strip():
        return "Please enter your question."
    try:
        retrieved_context = get_offline_context(question)
        if not retrieved_context:
            return "No offline resources available to answer this question."
        
        res = generate_qa_response(retrieved_context, question, language)
        save_to_cache(query_key, res, retrieved_context)
        return res
    except Exception as e:
        return f"Offline query error: {e}"

dina_prompt = """Namaste. You are a conversational, friendly, and expert Ayurvedic assistant. Below are details about an individual.
Generate a highly personalized, hour-by-hour Ayurvedic daily routine (Dinacharya) for them, which is great for long-term habit building.

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Data:
- Dominant Dosha: {dosha}
- Current Season: {season}
- Usual Wake-up Time: {wake_time}
- Occupation/Job: {occupation}
- Location: {location}

Your task is to:
1. Provide a step-by-step, hour-by-hour daily routine (Dinacharya) from morning to night.
2. Explain the Ayurvedic reasoning behind key suggestions.
3. Tailor the advice specifically for their Dosha, Season, Location, and fit it around their Occupation.

STRICT RULE: Do NOT provide modern generic advice. Focus exclusively on traditional Ayurvedic practices (e.g., oil pulling, tongue scraping, specific types of exercise, meal timings).

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use very short, simple sentences.
- Avoid technical terms or complex Sanskrit without explaining them simply.
- Make it extremely easy to understand.

### Response:
"""

def generate_dinacharya_plan(dosha, season, wake_time, occupation, location, language):
    import json
    query_key = f"DINA|{dosha}|{season}|{wake_time}|{occupation}|{location}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response'], data['context']

    if not dosha or not season or not wake_time or not occupation or not location:
        return "Please fill out all fields to generate your personalized hour-by-hour Dinacharya plan.", ""
        
    context = get_offline_context(f"Daily routine for {dosha} dosha in {season}")
    instruction = dina_prompt.format(dosha=dosha, season=season, wake_time=wake_time, occupation=occupation, location=location, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 600, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

herb_prompt = """Namaste. You are an expert Ayurvedic herbalist. Below is a user's health concern and their geographical region.
Provide 5–7 region-specific Ayurvedic herbs that naturally grow or are easily available in their area, along with a custom home remedy recipe.

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Data:
- Health Concern: {concern}
- Region / Location: {location}

Your task is to:
1. List 5-7 Ayurvedic herbs effective for this concern and available in or around their Region.
2. For EACH herb, mention its effect on the Doshas (Vata, Pitta, Kapha) and a simple preparation method.
3. Provide ONE custom home remedy recipe using some of these herbs, including step-by-step instructions.

STRICT RULE: Focus only on traditional, natural Ayurvedic methods. Do not recommend modern medicines. Keep safety in mind.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use clear, easy-to-understand language.
- Explain any Sanskrit terms.
- Use bullet points for readability.

### Response:
"""

def generate_herb_remedy(concern, location, language):
    import json
    query_key = f"HERB|{concern}|{location}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response'], data['context']

    if not concern or not location:
        return "Please provide both your health concern and location/region to find herbs.", ""
        
    context = get_offline_context(f"Ayurvedic herbs for {concern}")
    instruction = herb_prompt.format(concern=concern, location=location, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 600, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

ritu_prompt = """Namaste. You are an expert Ayurvedic practitioner. Below are details about an individual's Dosha, Season, and Region.
Generate a comprehensive Seasonal Wellness (Ritucharya) guide tailored specifically to them.

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Data:
- Dominant Dosha: {dosha}
- Current Season: {season}
- Region / Location: {location}

Your task is to:
1. Recommend foods to Eat and Foods to Avoid based on their Dosha and the Current Season.
2. Suggest region-specific seasonal herbs that are beneficial.
3. Advise on daily lifestyle rituals and exercise routines appropriate for this Dosha-Season combination.

STRICT RULE: Focus only on traditional, natural Ayurvedic methods. Do not provide modern generic advice. Keep safety in mind.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use clear, easy-to-understand language.
- Explain any Sanskrit terms simply.
- Use bullet points for readability.

### Response:
"""

def generate_ritucharya_guide(dosha, season, location, language):
    import json
    query_key = f"RITU|{dosha}|{season}|{location}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response'], data['context']

    if not dosha or not season or not location:
        return "Please provide your Dosha, Season, and Location to generate the Ritucharya guide.", ""
        
    context = get_offline_context(f"Ritucharya for {dosha} in {season}")
    instruction = ritu_prompt.format(dosha=dosha, season=season, location=location, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 600, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

yoga_prompt = """Namaste. You are an expert Ayurvedic Yoga and Pranayama instructor. Below are details about an individual's Dosha, fitness level, and age.
Generate a tailored Yoga, Pranayama, and Meditation routine suitable for them.

### Relevant Context (Traditional Knowledge):
{retrieved_context}

Data:
- Dominant Dosha: {dosha}
- Fitness Level: {fitness}
- Age: {age}

Your task is to:
1. Recommend a Dosha-specific Yoga sequence with both Sanskrit and English names for each pose.
2. Provide step-by-step instructions and duration for each pose, adjusted for their fitness level and age.
3. Suggest an appropriate Pranayama (breathing exercise) and Meditation technique suitable for them.

STRICT RULE: Focus only on traditional, natural Ayurvedic methods. Do not provide modern generic workout advice. Keep safety in mind.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use clear, easy-to-understand language.
- Explain any Sanskrit terms simply.
- Use bullet points for readability.

### Response:
"""

def generate_yoga_guide(dosha, fitness, age, language):
    import json
    query_key = f"YOGA|{dosha}|{fitness}|{age}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response'], data['context']

    if not dosha or not fitness or not age:
        return "Please provide your Dosha, Fitness Level, and Age to generate the Yoga guide.", ""
        
    context = get_offline_context(f"Yoga for {dosha} dosha")
    instruction = yoga_prompt.format(dosha=dosha, fitness=fitness, age=age, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 600, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

drug_prompt = """Namaste. You are an expert Ayurvedic doctor and pharmacologist. Analyze the side effects of this modern medicine.

### Relevant Context:
{retrieved_context}

Data:
- Medicine/Drug Name: {drug}

Your task is to structure your response EXACTLY like this (use bullet points and bold text for absolute clarity):

### ⚠️ Common Side Effects
- **Main Effect:** [List the most common side effect]
- **Secondary Effects:** [List 1-2 other side effects]

### 🎯 Affected Organs
- **Primary Organ:** [Name the organ]
- **Secondary Systems:** [Name any other affected bodily systems]

### 🔥 Ayurvedic Impact 
- **Dosha Imbalance:** [Explain if it increases Vata, Pitta, or Kapha]
- **Agni (Digestion):** [Explain how it affects digestive fire]

### 🌿 Mitigation Strategies
- **Dietary Change:** [1 specific food to eat or avoid]
- **Lifestyle Ritual:** [1 specific daily habit]
- **Herbal Support:** [1 specific Ayurvedic herb to protect the body]

STRICT RULE: Do NOT advise stopping medication. Always advise consulting a physician. Keep answers very concise.
LANGUAGE RULE: Output entirely in {language}.

### Response:
"""

def generate_drug_analysis(drug, language):
    import json
    query_key = f"DRUG|{drug}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        return data['response']

    if not drug.strip():
        return "Please provide the name of the medicine or drug."
        
    context = get_offline_context(f"Side effects and Ayurvedic impact of {drug}")
    instruction = drug_prompt.format(drug=drug, language=language, retrieved_context=context)
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 600, 
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_res = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    save_to_cache(query_key, final_res, context)
    return final_res, context

master_prompt = """Namaste. You are a Master Ayurvedic practitioner. A user has provided a complete profile of themselves. Generate a comprehensive, all-in-one personalized holistic Ayurvedic plan.

### Relevant Context (Traditional Knowledge):
{retrieved_context}

User Profile:
- Main Concern/Symptoms: {symptoms}
- Age: {age}, Fitness Level: {fitness}
- Location / Region: {location}, Current Season: {season}
- Diet: {diet}, Digestion: {digestion}, Sleep Pattern: {sleep}
- Occupation / Daily routine: {occupation}

Your task is to generate a perfectly structured Master Plan divided into exactly 5 separate sections.
CRITICAL RULE: You MUST start every single section with "### " followed by a number, exactly like the format below:

### 1. Dosha Analysis
[Briefly explain their likely Dosha imbalance]

### 2. Herbal Remedies
[2-3 specific home remedies native to {location} for their main concern]

### 3. Dietary Guidelines
[Foods to favor and avoid based on a {diet} diet in the {season} season]

### 4. Daily Routine
[3-4 lifestyle habits customized to fit around their {occupation}]

### 5. Yoga and Breathwork
[A short routine suitable for a {age}-year-old with a {fitness} fitness level]

STRICT RULE: Focus only on traditional, natural Ayurvedic methods. Keep safety in mind.

LANGUAGE RULE: Your entire response MUST be written in {language}. Do not strictly use English. If {language} is not English, accurately translate your output into {language}.

SIMPLICITY RULES:
- Use clear, easy-to-understand language.
- Explain any Sanskrit terms simply.

### Response:
"""

def generate_holistic_plan(symptoms, age, fitness, location, season, diet, digestion, sleep, occupation, language):
    import re, json
    # Composite key for Master Plan cache
    query_key = f"PLAN|{symptoms}|{age}|{fitness}|{location}|{season}|{diet}|{digestion}|{sleep}|{occupation}|{language}"
    cached = check_cache(query_key)
    if cached:
        data = json.loads(cached['data'])
        r = data['response']
        return r[0], r[1], r[2], r[3], r[4]

    if not symptoms.strip() or not location.strip():
        return ("Please provide symptoms and location.",)*5
        
    context = get_offline_context(f"Holistic plan for {symptoms} at {location} in {season}")
    instruction = master_prompt.format(
        symptoms=symptoms, age=age, fitness=fitness, location=location, 
        season=season, diet=diet, digestion=digestion, sleep=sleep, 
        occupation=occupation, language=language, retrieved_context=context
    )
    
    inputs = tokenizer([instruction], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs, 
        max_new_tokens = 1800, # Increased for full plan
        use_cache = True,
        temperature = 0.3, # Hallucination filter
        top_p = 0.85,
        repetition_penalty = 1.15
    )
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    res_split = response.split("### Response:")
    final_text = apply_hallucination_filters(res_split[-1].strip() if len(res_split) > 1 else response.strip())
    
    matches = list(re.finditer(r'###\s*\d+\.?[^\n]*', final_text))
    cleaned_parts = []
    
    if not matches:
        cleaned_parts = [final_text]
    else:
        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i+1].start() if i + 1 < len(matches) else len(final_text)
            cleaned_parts.append(final_text[start:end].strip())
    
    while len(cleaned_parts) < 5:
        cleaned_parts.append("Additional details requested but not generated. Please specify symptoms further.")
    
    # Cache the result
    save_to_cache(query_key, cleaned_parts[:5], context)
        
    return cleaned_parts[0], cleaned_parts[1], cleaned_parts[2], cleaned_parts[3], cleaned_parts[4]

ui_translations = {
    "English": {
        "symptoms_label": "Describe your symptoms / concerns",
        "symptoms_place": "E.g., I've been feeling very dry, with joint cracking, and I have trouble sleeping...",
        "submit_btn": "Get Ayurvedic Insight",
        "output_label": "AyurSLM's Suggestion",
        "prakriti_desc": "Select the options that best represent your *permanent, long-term nature*, not a temporary illness.",
        "body_label": "Body type",
        "digestion_label": "Digestion pattern",
        "sleep_label": "Sleep pattern",
        "weather_label": "Weather preference",
        "prakriti_btn": "Determine My Prakriti",
        "prakriti_out": "Your Prakriti & Lifestyle Tips",
        "qa_desc": "Paste an Ayurvedic text excerpt here and ask the AI a question about it. It will answer strictly based on your provided text.",
        "qa_context_lbl": "Ayurvedic Context",
        "qa_context_pl": "Paste the reference text or book excerpt here...",
        "qa_question_lbl": "Your Question",
        "qa_question_pl": "What does the text say about...",
        "qa_btn": "Find Answer",
        "qa_out": "Answer",
        "dina_desc": "Generates a personalized, hour-by-hour Ayurvedic daily routine based on your Dosha, occupation, location, wake-up time, and season. Great for long-term habit building.",
        "dina_dosha_lbl": "Dominant Dosha",
        "dina_season_lbl": "Current Season",
        "dina_wake_lbl": "Usual Wake-up Time",
        "dina_occ_lbl": "Occupation / Job routine",
        "dina_loc_lbl": "Location",
        "dina_btn": "Generate Dinacharya Plan",
        "dina_out": "Your Hour-by-Hour Daily Routine",
        "herb_desc": "Enter any health concern and get region-specific Ayurvedic herbs, Dosha effects, and a custom home remedy recipe.",
        "herb_concern_lbl": "Health Concern (e.g., hair fall, digestion, stress)",
        "herb_loc_lbl": "Your Region / Location",
        "herb_btn": "Find Herbs & Remedies",
        "herb_out": "Ayurvedic Herbs & Custom Remedy",
        "ritu_desc": "Generates a seasonal guide covering what foods to eat/avoid, seasonal herbs, lifestyle rituals, and exercise recommendations based on the Dosha and season.",
        "ritu_dosha_lbl": "Dominant Dosha",
        "ritu_season_lbl": "Current Season",
        "ritu_loc_lbl": "Your Region / Location",
        "ritu_btn": "Generate Seasonal Guide (Ritucharya)",
        "ritu_out": "Your Seasonal Wellness Guide",
        "yoga_desc": "Get a personalized Yoga, Pranayama, and Meditation sequence based on your Dosha, fitness level, and age.",
        "yoga_dosha_lbl": "Dominant Dosha",
        "yoga_fitness_lbl": "Fitness Level",
        "yoga_age_lbl": "Age",
        "yoga_btn": "Generate Yoga Routine",
        "yoga_out": "Your Custom Yoga & Pranayama Plan",
        "drug_desc": "Enter the name of a modern medicine to understand its side effects, its impact on your Doshas, and Ayurvedic ways to support your body while taking it.",
        "drug_name_lbl": "Medicine / Drug Name",
        "drug_btn": "Analyze Medicine",
        "drug_out": "Side Effects & Ayurvedic Analysis",
        "master_desc": "Fill out your complete profile once to instantly generate a comprehensive Ayurvedic Master Plan covering Dosha analysis, diet, herbs, daily routine, and yoga.",
        "master_sym_lbl": "Main Concern / Symptoms",
        "master_btn": "Generate Complete Master Plan",
        "m_out_1": "1. Dosha Analysis",
        "m_out_2": "2. Herbal Remedies",
        "m_out_3": "3. Dietary Guidelines",
        "m_out_4": "4. Daily Routine (Dinacharya)",
        "m_out_5": "5. Yoga & Breathwork"
    },
    "Kannada (ಕನ್ನಡ)": {
        "symptoms_label": "ನಿಮ್ಮ ಲಕ್ಷಣಗಳು / ಕಾಳಜಿಗಳನ್ನು ವಿವರಿಸಿ",
        "symptoms_place": "ಉದಾ: ನನಗೆ ತುಂಬಾ ಒಣಗಿದ ಅನುಭವ, ಕೀಲುಗಳಲ್ಲಿ ನೋವು, ಮತ್ತು ನಿದ್ರೆ ಬರುತ್ತಿಲ್ಲ...",
        "submit_btn": "ಆಯುರ್ವೇದ ಸಲಹೆ ಪಡೆಯಿರಿ",
        "output_label": "AyurSLM ಸಲಹೆ",
        "prakriti_desc": "ನಿಮ್ಮ *ಶಾಶ್ವತ, ದೀರ್ಘಕಾಲೀನ ಪ್ರಕೃತಿಯನ್ನು* ಉತ್ತಮವಾಗಿ ಪ್ರತಿನಿಧಿಸುವ ಆಯ್ಕೆಗಳನ್ನು ಆರಿಸಿ.",
        "body_label": "ದೇಹದ ಪ್ರಕಾರ",
        "digestion_label": "ಜೀರ್ಣಕ್ರಿಯೆಯ ಮಾದರಿ",
        "sleep_label": "ನಿದ್ರೆಯ ಮಾದರಿ",
        "weather_label": "ಹವಾಮಾನ ಆದ್ಯತೆ",
        "prakriti_btn": "ನನ್ನ ಪ್ರಕೃತಿ ನಿರ್ಧರಿಸಿ",
        "prakriti_out": "ನಿಮ್ಮ ಪ್ರಕೃತಿ ಮತ್ತು ಜೀವನಶೈಲಿ ಸಲಹೆಗಳು",
        "qa_desc": "ಆಯುರ್ವೇದ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಅಂಟಿಸಿ ಮತ್ತು AI ಗೆ ಅದರ ಬಗ್ಗೆ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ.",
        "qa_context_lbl": "ಆಯುರ್ವೇದ ಸಂದರ್ಭ",
        "qa_context_pl": "ಉಲ್ಲೇಖ ಪಠ್ಯ ಅಥವಾ ಪುಸ್ತಕದ ತುಣುಕನ್ನು ಇಲ್ಲಿ ಅಂಟಿಸಿ...",
        "qa_question_lbl": "ನಿಮ್ಮ ಪ್ರಶ್ನೆ",
        "qa_question_pl": "ಪಠ್ಯವು ಇದರ ಬಗ್ಗೆ ಏನು ಹೇಳುತ್ತದೆ...",
        "qa_btn": "ಉತ್ತರ ಹುಡುಕಿ",
        "qa_out": "ಉತ್ತರ",
        "dina_desc": "ನಿಮ್ಮ ದೋಷ, ವೃತ್ತಿ, ಸ್ಥಳ, ಎಚ್ಚರಗೊಳ್ಳುವ ಸಮಯ ಮತ್ತು ಋತುವಿನ ಆಧಾರದ ಮೇಲೆ ವೈಯಕ್ತಿಕಗೊಳಿಸಿದ, ಗಂಟೆಗೊಮ್ಮೆ ಆಯುರ್ವೇದ ದೈನಂದಿನ ದಿನಚರಿಯನ್ನು ರಚಿಸುತ್ತದೆ. ದೀರ್ಘಕಾಲೀನ ಅಭ್ಯಾಸ ನಿರ್ಮಾಣಕ್ಕೆ ಉತ್ತಮವಾಗಿದೆ.",
        "dina_dosha_lbl": "ಪ್ರಧಾನ ದೋಷ",
        "dina_season_lbl": "ಪ್ರಸ್ತುತ ಋತು",
        "dina_wake_lbl": "ಎಚ್ಚರಗೊಳ್ಳುವ ಸಾಮಾನ್ಯ ಸಮಯ",
        "dina_occ_lbl": "ವೃತ್ತಿ / ಉದ್ಯೋಗದ ದಿನಚರಿ",
        "dina_loc_lbl": "ಸ್ಥಳ",
        "dina_btn": "ದಿನಚರ್ಯೆ ಯೋಜನೆಯನ್ನು ರಚಿಸಿ",
        "dina_out": "ನಿಮ್ಮ ಗಂಟೆ-ಗಂಟೆಯ ದೈನಂದಿನ ದಿನಚರಿ",
        "herb_desc": "ಯಾವುದೇ ಆರೋಗ್ಯ ಕಾಳಜಿಯನ್ನು ನಮೂದಿಸಿ ಮತ್ತು ಪ್ರದೇಶ-ನಿರ್ದಿಷ್ಟ ಆಯುರ್ವೇದ ಗಿಡಮೂಲಿಕೆಗಳು ಮತ್ತು ಕಸ್ಟಮ್ ಮನೆಮದ್ದು ಪಾಕವಿಧಾನವನ್ನು ಪಡೆಯಿರಿ.",
        "herb_concern_lbl": "ಆರೋಗ್ಯದ ಕಾಳಜಿ (ಉದಾ., ಕೂದಲು ಉದುರುವಿಕೆ, ಜೀರ್ಣಕ್ರಿಯೆ, ಒತ್ತಡ)",
        "herb_loc_lbl": "ನಿಮ್ಮ ಪ್ರದೇಶ / ಸ್ಥಳ",
        "herb_btn": "ಗಿಡಮೂಲಿಕೆಗಳು ಮತ್ತು ಪರಿಹಾರಗಳನ್ನು ಹುಡುಕಿ",
        "herb_out": "ಆಯುರ್ವೇದ ಗಿಡಮೂಲಿಕೆಗಳು ಮತ್ತು ಕಸ್ಟಮ್ ಪರಿಹಾರ",
        "ritu_desc": "ದೋಷ ಮತ್ತು ಋತುವಿನ ಆಧಾರದ ಮೇಲೆ ಯಾವ ಆಹಾರವನ್ನು ಸೇವಿಸಬೇಕು/ತಪ್ಪಿಸಬೇಕು, ಋತುಮಾನದ ಗಿಡಮೂಲಿಕೆಗಳು, ಜೀವನಶೈಲಿ ಆಚರಣೆಗಳು ಮತ್ತು ವ್ಯಾಯಾಮದ ಶಿಫಾರಸುಗಳನ್ನು ಒಳಗೊಂಡ ಋತುಮಾನದ ಮಾರ್ಗದರ್ಶಿಯನ್ನು ರಚಿಸುತ್ತದೆ.",
        "ritu_dosha_lbl": "ಪ್ರಧಾನ ದೋಷ",
        "ritu_season_lbl": "ಪ್ರಸ್ತುತ ಋತು",
        "ritu_loc_lbl": "ನಿಮ್ಮ ಪ್ರದೇಶ / ಸ್ಥಳ",
        "ritu_btn": "ಋತುಮಾನದ ಮಾರ್ಗದರ್ಶಿ ರಚಿಸಿ",
        "ritu_out": "ನಿಮ್ಮ ಋತುಮಾನದ ಸ್ವಾಸ್ಥ್ಯ ಮಾರ್ಗದರ್ಶಿ",
        "yoga_desc": "ನಿಮ್ಮ ದೋಷ, ಫಿಟ್‌ನೆಸ್ ಮಟ್ಟ ಮತ್ತು ವಯಸ್ಸಿನ ಆಧಾರದ ಮೇಲೆ ವೈಯಕ್ತೀಕರಿಸಿದ ಯೋಗ, ಪ್ರಾಣಾಯಾಮ ಮತ್ತು ಧ್ಯಾನದ ಅನುಕ್ರಮವನ್ನು ಪಡೆಯಿರಿ.",
        "yoga_dosha_lbl": "ಪ್ರಧಾನ ದೋಷ",
        "yoga_fitness_lbl": "ಫಿಟ್‌ನೆಸ್ ಮಟ್ಟ",
        "yoga_age_lbl": "ವಯಸ್ಸು",
        "yoga_btn": "ಯೋಗ ದಿನಚರಿಯನ್ನು ರಚಿಸಿ",
        "yoga_out": "ನಿಮ್ಮ ಕಸ್ಟಮ್ ಯೋಗ ಮತ್ತು ಪ್ರಾಣಾಯಾಮ ಯೋಜನೆ",
        "drug_desc": "ಆಧುನಿಕ ಔಷಧಿಯ ಅಡ್ಡಪರಿಣಾಮಗಳು, ಜೀರ್ಣಕ್ರಿಯೆಯ ಮೇಲಿನ ಪ್ರಭಾವ ಮತ್ತು ಆಯುರ್ವೇದ ಪರಿಹಾರಗಳನ್ನು ತಿಳಿಯಲು ಔಷಧಿಯ ಹೆಸರನ್ನು ನಮೂದಿಸಿ.",
        "drug_name_lbl": "ಔಷಧಿ / ಡ್ರಗ್ ಹೆಸರು",
        "drug_btn": "ಔಷಧಿಯನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
        "drug_out": "ಅಡ್ಡಪರಿಣಾಮಗಳು ಮತ್ತು ಆಯುರ್ವೇದ ವಿಶ್ಲೇಷಣೆ",
        "master_desc": "ನಿಮ್ಮ ಸಂಪೂರ್ಣ ವಿವರಗಳನ್ನು ಒಮ್ಮೆ ಭರ್ತಿ ಮಾಡಿ ಮತ್ತು ತಕ್ಷಣವೇ ದೋಷ ವಿಶ್ಲೇಷಣೆ, ಆಹಾರ, ಗಿಡಮೂಲಿಕೆಗಳು, ದೈನಂದಿನ ದಿನಚರಿ ಮತ್ತು ಯೋಗವನ್ನು ಒಳಗೊಂಡಿರುವ ಸಮಗ್ರ ಆಯುರ್ವೇದ ಮಾಸ್ಟರ್ ಯೋಜನೆಯನ್ನು ಪಡೆಯಿರಿ.",
        "master_sym_lbl": "ಮುಖ್ಯ ಕಾಳಜಿ / ಲಕ್ಷಣಗಳು",
        "master_btn": "ಸಂಪೂರ್ಣ ಮಾಸ್ಟರ್ ಯೋಜನೆ ರಚಿಸಿ",
        "m_out_1": "1. ದೋಷ ವಿಶ್ಲೇಷಣೆ",
        "m_out_2": "2. ಗಿಡಮೂಲಿಕೆ ಪರಿಹಾರಗಳು",
        "m_out_3": "3. ಆಹಾರ ಮಾರ್ಗಸೂಚಿಗಳು",
        "m_out_4": "4. ದೈನಂದಿನ ದಿನಚರಿ",
        "m_out_5": "5. ಯೋಗ ಮತ್ತು ಉಸಿರಾಟ"
    },
    "Hindi (हिंदी)": {
        "symptoms_label": "अपने लक्षणों / चिंताओं का वर्णन करें",
        "symptoms_place": "उदा., मुझे बहुत सूखापन महसूस हो रहा है, जोड़ों में दर्द है...",
        "submit_btn": "आयुर्वेदिक सलाह प्राप्त करें",
        "output_label": "AyurSLM का सुझाव",
        "prakriti_desc": "वे विकल्प चुनें जो आपकी *स्थायी, दीर्घकालिक प्रकृति* का प्रतिनिधित्व करते हों।",
        "body_label": "शरीर का प्रकार",
        "digestion_label": "पाचन प्रतिमान",
        "sleep_label": "नींद का प्रतिमान",
        "weather_label": "मौसम की पसंद",
        "prakriti_btn": "मेरी प्रकृति निर्धारित करें",
        "prakriti_out": "आपकी प्रकृति और जीवनशैली सुझाव",
        "qa_desc": "आयुर्वेदिक संदर्भ पाठ यहाँ चिपकाएँ और AI से एक प्रश्न पूछें।",
        "qa_context_lbl": "आयुर्वेदिक संदर्भ",
        "qa_context_pl": "संदर्भ पाठ या पुस्तक का अंश यहाँ चिपकाएँ...",
        "qa_question_lbl": "आपका प्रश्न",
        "qa_question_pl": "पाठ क्या कहता है इसके बारे में...",
        "qa_btn": "उत्तर खोजें",
        "qa_out": "उत्तर",
        "dina_desc": "आपके दोष, पेशे, स्थान, जागने के समय और मौसम के आधार पर एक व्यक्तिगत, घंटे-दर-घंटे आयुर्वेदिक दैनिक दिनचर्या उत्पन्न करता है। दीर्घकालिक आदत निर्माण के लिए बहुत अच्छा।",
        "dina_dosha_lbl": "प्रमुख दोष",
        "dina_season_lbl": "वर्तमान मौसम",
        "dina_wake_lbl": "जागने का सामान्य समय",
        "dina_occ_lbl": "पेशा / कार्य दिनचर्या",
        "dina_loc_lbl": "स्थान",
        "dina_btn": "दिनचर्या योजना बनाएँ",
        "dina_out": "आपकी घंटे-दर-घंटे दैनिक दिनचर्या",
        "herb_desc": "कोई भी स्वास्थ्य चिंता दर्ज करें और क्षेत्र-विशिष्ट आयुर्वेदिक जड़ी-बूटियाँ और एक कस्टम घरेलू उपचार नुस्खा प्राप्त करें।",
        "herb_concern_lbl": "स्वास्थ्य संबंधी चिंता (उदा., बालों का झड़ना, पाचन, तनाव)",
        "herb_loc_lbl": "आपका क्षेत्र / स्थान",
        "herb_btn": "जड़ी-बूटियाँ और उपचार खोजें",
        "herb_out": "आयुर्वेदिक जड़ी-बूटियाँ और कस्टम उपचार",
        "ritu_desc": "दोष और मौसम के आधार पर खाने/बचने वाले खाद्य पदार्थों, मौसमी जड़ी-बूटियों, जीवनशैली अनुष्ठानों और व्यायाम की सिफारिशों को कवर करते हुए एक मौसमी गाइड उत्पन्न करता है।",
        "ritu_dosha_lbl": "प्रमुख दोष",
        "ritu_season_lbl": "वर्तमान मौसम",
        "ritu_loc_lbl": "आपका क्षेत्र / स्थान",
        "ritu_btn": "मौसमी गाइड बनाएँ (ऋतुचर्या)",
        "ritu_out": "आपका मौसमी कल्याण गाइड",
        "yoga_desc": "अपने दोष, फिटनेस स्तर और उम्र के आधार पर एक व्यक्तिगत योग, प्राणायाम और ध्यान अनुक्रम प्राप्त करें।",
        "yoga_dosha_lbl": "प्रमुख दोष",
        "yoga_fitness_lbl": "फिटनेस स्तर",
        "yoga_age_lbl": "आयु",
        "yoga_btn": "योग दिनचर्या बनाएँ",
        "yoga_out": "आपकी कस्टम योग और प्राणायाम योजना",
        "drug_desc": "किसी आधुनिक दवा के दुष्प्रभाव, आपके दोषों पर इसका प्रभाव, और इसे लेते समय आपके शरीर का समर्थन करने के आयुर्वेदिक तरीकों को समझने के लिए दवा का नाम दर्ज करें।",
        "drug_name_lbl": "दवा / औषधि का नाम",
        "drug_btn": "दवा का विश्लेषण करें",
        "drug_out": "दुष्प्रभाव और आयुर्वेदिक विश्लेषण",
        "master_desc": "दोष विश्लेषण, आहार, जड़ी-बूटियों, दैनिक दिनचर्या और योग को कवर करते हुए तुरंत एक व्यापक आयुर्वेदिक मास्टर योजना उत्पन्न करने के लिए एक बार अपनी पूरी प्रोफ़ाइल भरें।",
        "master_sym_lbl": "मुख्य चिंता / लक्षण",
        "master_btn": "संपूर्ण मास्टर योजना बनाएँ",
        "m_out_1": "1. दोष विश्लेषण",
        "m_out_2": "2. हर्बल उपचार",
        "m_out_3": "3. आहार दिशानिर्देश",
        "m_out_4": "4. दैनिक दिनचर्या",
        "m_out_5": "5. योग और प्राणायाम"
    },
    "Telugu (తెలుగు)": {
        "symptoms_label": "మీ లక్షణాలు / ఆందోళనలను వివరించండి",
        "symptoms_place": "ఉదా., నాకు చాలా పొడిగా అనిపిస్తుంది, కీళ్ల నొప్పులు...",
        "submit_btn": "ఆయుర్వేద సలహాను పొందండి",
        "output_label": "AyurSLM సూచన",
        "prakriti_desc": "మీ *శాశ్వత, దీర్ఘకాలిక ప్రకృతిని* ఉత్తమంగా సూచించే ఎంపికలను ఎంచుకోండి.",
        "body_label": "శరీర రకం",
        "digestion_label": "జీర్ణక్రమ నమూనా",
        "sleep_label": "నిద్ర నమూనా",
        "weather_label": "వాతావరణ ప్రాధాన్యత",
        "prakriti_btn": "నా ప్రకృతిని నిర్ణయించండి",
        "prakriti_out": "మీ ప్రకృతి & జీవనశైలి చిట్కాలు",
        "qa_desc": "ఆయుర్వేద వచనాన్ని ఇక్కడ అతికించి, దాని గురించి AI కి ఒక ప్రశ్న అడగండి.",
        "qa_context_lbl": "ఆయుర్వేద సందర్భం",
        "qa_context_pl": "సూచన వచనం లేదా పుస్తక భాగాన్ని ఇక్కడ అతికించండి...",
        "qa_question_lbl": "మీ ప్రశ్న",
        "qa_question_pl": "వచనం దేని గురించి చెబుతుంది...",
        "qa_btn": "సమాధానం కనుగొనండి",
        "qa_out": "సమాధానం",
        "dina_desc": "మీ దోషం, వృత్తి, స్థానం, మేల్కొనే సమయం మరియు రుతువు ఆధారంగా వ్యక్తిగతీకరించిన, గంట-గంట ఆయుర్వేద రోజువారీ దినచర్యను రూపొందిస్తుంది. దీర్ఘకాలిక అలవాట్ల నిర్మాణానికి గొప్పది.",
        "dina_dosha_lbl": "ప్రధాన దోషం",
        "dina_season_lbl": "ప్రస్తుత రుతువు",
        "dina_wake_lbl": "సాధారణంగా మేల్కొనే సమయం",
        "dina_occ_lbl": "వృత్తి / ఉద్యోగ దినచర్య",
        "dina_loc_lbl": "స్థానం",
        "dina_btn": "దినచర్య ప్రణాళికను రూపొందించండి",
        "dina_out": "మీ గంట-గంట దినచర్య",
        "herb_desc": "ఏదైనా ఆరోగ్య సమస్యను నమోదు చేయండి మరియు ప్రాంత-నిర్దిష్ట ఆయుర్వేద మూలికలు మరియు అనుకూల గృహ నివారణ రెసిపీని పొందండి.",
        "herb_concern_lbl": "ఆరోగ్య సమస్య (ఉదా., జుట్టు రాలడం, జీర్ణక్రియ, ఒత్తిడి)",
        "herb_loc_lbl": "మీ ప్రాంతం / స్థానం",
        "herb_btn": "మూలికలు మరియు నివారణలను కనుగొనండి",
        "herb_out": "ఆయుర్వేద మూలికలు మరియు అనుకూల నివారణ",
        "ritu_desc": "దోషం మరియు రుతువు ఆధారంగా తినాల్చిన/తప్పించవలసిన ఆహారాలు, కాలానుగుణ మూలికలు, జీవనశైలి ఆచారాలు మరియు వ్యాయామ సిఫార్సులతో కూడిన కాలానుగుణ మార్గదర్శిని రూపొందిస్తుంది.",
        "ritu_dosha_lbl": "ప్రధాన దోషం",
        "ritu_season_lbl": "ప్రస్తుత రుతువు",
        "ritu_loc_lbl": "మీ ప్రాంతం / స్థానం",
        "ritu_btn": "కాలానుగుణ మార్గదర్శిని రూపొందించండి (రుతుచర్య)",
        "ritu_out": "మీ కాలానుగుణ ఆరోగ్య మార్గదర్శి",
        "yoga_desc": "మీ దోషం, ఫిట్‌నెస్ స్థాయి మరియు వయస్సు ఆధారంగా వ్యక్తిగతీకరించిన యోగా, ప్రాణాయామం మరియు ధ్యాన క్రమాన్ని పొందండి.",
        "yoga_dosha_lbl": "ప్రధాన దోషం",
        "yoga_fitness_lbl": "ఫిట్‌నెస్ స్థాయి",
        "yoga_age_lbl": "వయస్సు",
        "yoga_btn": "యోగా దినచర్యను రూపొందించండి",
        "yoga_out": "మీ కస్టమ్ యోగా & ప్రాణాయామ ప్రణాళిక",
        "drug_desc": "ఆధునిక ఔషధం యొక్క దుష్ప్రభావాలు, మీ దోషాలపై దాని ప్రభావం మరియు ఆయుర్వేద మార్గాలను అర్థం చేసుకోవడానికి ఔషధం పేరును నమోదు చేయండి.",
        "drug_name_lbl": "మందు / ఔషధం పేరు",
        "drug_btn": "ఔషధాన్ని విశ్లేషించండి",
        "drug_out": "దుష్ప్రభావాలు & ఆయుర్వేద విశ్లేషణ",
        "master_desc": "దోష విశ్లేషణ, ఆహారం, మూలికలు, రోజువారీ దినచర్య మరియు యోగాను కవర్ చేసే సమగ్ర ఆయుర్వేద మాస్టర్ ప్లాన్‌ను తక్షణమే రూపొందించడానికి మీ పూర్తి ప్రొఫైల్‌ను ఒకసారి పూరించండి.",
        "master_sym_lbl": "ప్రధాన ఆందోళన / లక్షణాలు",
        "master_btn": "పూర్తి మాస్టర్ ప్లాన్‌ను రూపొందించండి",
        "m_out_1": "1. దోష విశ్లేషణ",
        "m_out_2": "2. మూలికా నివారణలు",
        "m_out_3": "3. ఆహార మార్గదర్శకాలు",
        "m_out_4": "4. రోజువారీ దినచర్య",
        "m_out_5": "5. యోగా & శ్వాస వ్యాయామాలు"
    },
    "Tamil (தமிழ்)": {
        "symptoms_label": "உங்கள் அறிகுறிகள் / கவலைகளை விவரிக்கவும்",
        "symptoms_place": "உதா., எனக்கு மிகவும் வறட்சியாக இருக்கிறது, மூட்டு வலி...",
        "submit_btn": "ஆயுர்வேத ஆலோசனையைப் பெறுங்கள்",
        "output_label": "AyurSLM பரிந்துரை",
        "prakriti_desc": "உங்கள் உண்மையான, நீண்ட கால தன்மையை சிறப்பாகக் குறிக்கும் விருப்பத்தைத் தேர்வு செய்யவும்.",
        "body_label": "உடல் வகை",
        "digestion_label": "செரிமான முறை",
        "sleep_label": "தூக்க முறை",
        "weather_label": "வானிலை விருப்பம்",
        "prakriti_btn": "எனது பிரகிருதியை தீர்மானிக்கவும்",
        "prakriti_out": "உங்கள் பிரகிருதி மற்றும் வாழ்க்கை முறை குறிப்புகள்",
        "qa_desc": "ஆயுர்வேத உரையை இங்கே ஒட்டி, அதைப் பற்றி AI-யைக் கேளுங்கள்.",
        "qa_context_lbl": "ஆயுர்வேத சூழல்",
        "qa_context_pl": "குறிப்பு உரை அல்லது புத்தகப் பகுதியை இங்கே ஒட்டவும்...",
        "qa_question_lbl": "உங்கள் கேள்வி",
        "qa_question_pl": "உரை இதைப் பற்றி என்ன சொல்கிறது...",
        "qa_btn": "பதிலைக் கண்டுபிடி",
        "qa_out": "பதில்",
        "dina_desc": "உங்கள் தோஷம், தொழில், இருப்பிடம், எழுந்திருக்கும் நேரம் மற்றும் பருவத்தின் அடிப்படையில் தனிப்பயனாக்கப்பட்ட, மணிநேர ஆயுர்வேத தினசரி வழக்கத்தை உருவாக்குகிறது.",
        "dina_dosha_lbl": "பிரதான தோஷம்",
        "dina_season_lbl": "தற்போதைய பருவம்",
        "dina_wake_lbl": "பொதுவாக எழுந்திருக்கும் நேரம்",
        "dina_occ_lbl": "தொழில் / வேலை வழக்கம்",
        "dina_loc_lbl": "இடம்",
        "dina_btn": "தினச்சரித் திட்டத்தை உருவாக்கவும்",
        "dina_out": "உங்கள் மணிநேர தினசரி வழக்கம்",
        "herb_desc": "எந்தவொரு உடல்நலக் கவலையையும் உள்ளிட்டு, தயாரிப்பு முறைகள், தோஷ விளைவுகள் மற்றும் தனிப்பயன் வீட்டு வைத்திய செய்முறையுடன் பிராந்திய அளவிலான ஆயுர்வேத மூலிகைகளைப் பெறுங்கள்.",
        "herb_concern_lbl": "உடல்நலக் கவலை (உதா., முடி உதிர்தல், செரிமானம், மன அழுத்தம்)",
        "herb_loc_lbl": "உங்கள் பகுதி / இடம்",
        "herb_btn": "மூலிகைகள் மற்றும் வைத்தியங்களைக் கண்டறியவும்",
        "herb_out": "ஆயுர்வேத மூலிகைகள் & தனிப்பயன் வைத்தியம்",
        "ritu_desc": "தோஷம் மற்றும் பருவத்தின் அடிப்படையில் உண்ண வேண்டிய/தவிர்க்க வேண்டிய உணவுகள், பருவகால மூலிகைகள், வாழ்க்கை முறை சடங்குகள் மற்றும் உடற்பயிற்சி பரிந்துரைகளை உள்ளடக்கிய பருவகால வழிகாட்டியை உருவாக்குகிறது.",
        "ritu_dosha_lbl": "பிரதான தோஷம்",
        "ritu_season_lbl": "தற்போதைய பருவம்",
        "ritu_loc_lbl": "உங்கள் பகுதி / இடம்",
        "ritu_btn": "பருவகால வழிகாட்டியை உருவாக்கவும்",
        "ritu_out": "உங்கள் பருவகால ஆரோக்கிய வழிகாட்டி",
        "yoga_desc": "உங்கள் தோஷம், உடற்தகுதி நிலை மற்றும் வயதின் அடிப்படையில் தனிப்பயனாக்கப்பட்ட யோகா, பிராணயாமா மற்றும் தியான வரிசையைப் பெறுங்கள்.",
        "yoga_dosha_lbl": "பிரதான தோஷம்",
        "yoga_fitness_lbl": "உடற்தகுதி நிலை",
        "yoga_age_lbl": "வயது",
        "yoga_btn": "யோகா வழக்கத்தை உருவாக்கவும்",
        "yoga_out": "உங்கள் தனிப்பயன் யோகா மற்றும் பிராணயாமா திட்டம்",
        "drug_desc": "நவீன மருந்தின் பக்க விளைவுகள், உங்கள் தோஷங்களில் அதன் தாக்கம் மற்றும் ஆயுர்வேத வழிகளைப் புரிந்துகொள்ள மருந்தின் பெயரை உள்ளிடவும்.",
        "drug_name_lbl": "மருந்து / மாத்திரை பெயர்",
        "drug_btn": "மருந்தை பகுப்பாய்வு செய்",
        "drug_out": "பக்க விளைவுகள் & ஆயுர்வேத பகுப்பாய்வு",
        "master_desc": "தோஷ பகுப்பாய்வு, உணவு, மூலிகைகள், தினசரி வழக்கம் மற்றும் யோகாவை உள்ளடக்கிய விரிவான ஆயுர்வேத மாஸ்டர் திட்டத்தை உடனடியாக உருவாக்க உங்கள் முழு சுயவிவரத்தையும் ஒரு முறை நிரப்பவும்.",
        "master_sym_lbl": "முக்கிய கவலை / அறிகுறிகள்",
        "master_btn": "முழுமையான மாஸ்டர் திட்டத்தை உருவாக்கு",
        "m_out_1": "1. தோஷ பகுப்பாய்வு",
        "m_out_2": "2. மூலிகை வைத்தியம்",
        "m_out_3": "3. உணவு வழிகாட்டுதல்கள்",
        "m_out_4": "4. தினசரி வழக்கம்",
        "m_out_5": "5. யோகா & மூச்சுப் பயிற்சி"
    },
    "Marathi (मराठी)": {
        "symptoms_label": "आपल्या लक्षणांचे / समस्यांचे वर्णन करा",
        "symptoms_place": "उदा., मला खूप कोरडे वाटत आहे, सांधे दुखत आहेत...",
        "submit_btn": "आयुर्वेदिक सल्ला मिळवा",
        "output_label": "AyurSLM चा सल्ला",
        "prakriti_desc": "अशी पर्याये निवडा जी तुमच्या *कायमस्वरूपी, दीर्घकालिक प्रकृतीचे* प्रतिनिधित्व करतात.",
        "body_label": "शरीराचा प्रकार",
        "digestion_label": "पचन पद्धत",
        "sleep_label": "झोपेची पद्धत",
        "weather_label": "हवामानाची पसंती",
        "prakriti_btn": "माझी प्रकृती ठरवा",
        "prakriti_out": "तुमची प्रकृती आणि जीवनशैली टिप्स",
        "qa_desc": "आयुर्वेदिक मजकूर येथे पेस्ट करा आणि त्याबद्दल AI ला एक प्रश्न विचारा.",
        "qa_context_lbl": "आयुर्वेदिक संदर्भ",
        "qa_context_pl": "संदर्भ मजकूर किंवा पुस्तकाचा अंश येथे पेस्ट करा...",
        "qa_question_lbl": "तुमचा प्रश्न",
        "qa_question_pl": "मजकूर याबद्दल काय सांगतो...",
        "qa_btn": "उत्तर शोधा",
        "qa_out": "उत्तर",
        "dina_desc": "तुमचा दोष, व्यवसाय, स्थान, उठण्याची वेळ आणि ऋतू यावर आधारित वैयक्तिकृत, तासांनुसार आयुर्वेदिक दैनिक दिनचर्या तयार करते. दीर्घकालीन सवयीसाठी उत्तम.",
        "dina_dosha_lbl": "प्रबळ दोष",
        "dina_season_lbl": "सध्याचा ऋतू",
        "dina_wake_lbl": "उठण्याची सामान्य वेळ",
        "dina_occ_lbl": "व्यवसाय / नोकरीची दिनचर्या",
        "dina_loc_lbl": "स्थान",
        "dina_btn": "दिनचर्या योजना तयार करा",
        "dina_out": "तुमची तासांनुसार दैनिक दिनचर्या",
        "herb_desc": "कोणतीही आरोग्य समस्या प्रविष्ट करा आणि तयारीच्या पद्धती, दोष प्रभाव आणि सानुकूल घरगुती उपाय रेसिपीसह प्रदेश-विशिष्ट आयुर्वेदिक औषधी वनस्पती मिळवा.",
        "herb_concern_lbl": "आरोग्य समस्या (उदा., केस गळणे, पचन, तणाव)",
        "herb_loc_lbl": "तुमचा प्रदेश / स्थान",
        "herb_btn": "औषधी वनस्पती आणि उपाय शोधा",
        "herb_out": "आयुर्वेदिक औषधी वनस्पती आणि सानुकूल उपाय",
        "ritu_desc": "दोष आणि ऋतूवर आधारित काय खावे/टाळावे, हंगामी औषधी वनस्पती, जीवनशैली विधी आणि व्यायाम शिफारसींचा समावेश असलेले हंगामी मार्गदर्शक तयार करते.",
        "ritu_dosha_lbl": "प्रबळ दोष",
        "ritu_season_lbl": "सध्याचा ऋतू",
        "ritu_loc_lbl": "तुमचा प्रदेश / स्थान",
        "ritu_btn": "हंगामी मार्गदर्शक तयार करा (ऋतुचर्या)",
        "ritu_out": "तुमचे हंगामी कल्याण मार्गदर्शक",
        "yoga_desc": "तुमचा दोष, फिटनेस पातळी आणि वयावर आधारित वैयक्तिकृत योग, प्राणायाम आणि ध्यान अनुक्रम मिळवा.",
        "yoga_dosha_lbl": "प्रबळ दोष",
        "yoga_fitness_lbl": "फिटनेस पातळी",
        "yoga_age_lbl": "वय",
        "yoga_btn": "योग दिनचर्या तयार करा",
        "yoga_out": "तुमची सानुकूल योग आणि प्राणायाम योजना",
        "drug_desc": "आधुनिक औषधाचे दुष्परिणाम, तुमच्या दोषांवरील त्याचा प्रभाव आणि शरीराला आधार देण्याचे आयुर्वेदिक मार्ग समजून घेण्यासाठी औषधाचे नाव प्रविष्ट करा.",
        "drug_name_lbl": "औषध / गोळीचे नाव",
        "drug_btn": "औषधाचे विश्लेषण करा",
        "drug_out": "दुष्परिणाम आणि आयुर्वेदिक विश्लेषण",
        "master_desc": "दोष विश्लेषण, आहार, औषधी वनस्पती, दैनिक दिनचर्या आणि योग कव्हर करणारी सर्वसमावेशक आयुर्वेदिक मास्टर योजना त्वरित तयार करण्यासाठी आपली संपूर्ण प्रोफाइल एकदाच भरा.",
        "master_sym_lbl": "मुख्य चिंता / लक्षणे",
        "master_btn": "संपूर्ण मास्टर योजना तयार करा",
        "m_out_1": "1. दोष विश्लेषण",
        "m_out_2": "2. हर्बल उपाय",
        "m_out_3": "3. आहार मार्गदर्शक तत्त्वे",
        "m_out_4": "4. दैनिक दिनचर्या",
        "m_out_5": "5. योग आणि श्वासोच्छ्वास"
    }
}

# UI Theme and CSS Configuration
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Dynamic Animated Background */
@keyframes gradientAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
    background: linear-gradient(-45deg, #0f172a, #1e293b, #064e3b, #0f172a) !important;
    background-size: 400% 400% !important;
    animation: gradientAnimation 15s ease infinite !important;
    background-attachment: fixed !important;
    color: #f8fafc !important;
}

/* Premium Glassmorphic Blocks */
.gradio-container .gr-block, .gradio-container .gr-box, .gradio-container .gr-panel {
    background: rgba(15, 23, 42, 0.55) !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 24px !important;
    box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    position: relative;
}

.gradio-container .gr-block:hover {
    transform: translateY(-5px) scale(1.01) !important;
    box-shadow: 0 30px 60px -15px rgba(16, 185, 129, 0.25) !important;
    border-color: rgba(52, 211, 153, 0.3) !important;
}

/* Header Glow & Animations */
.header-box {
    text-align: center;
    padding: 5rem 2rem;
    margin-bottom: 3rem;
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, rgba(15, 23, 42, 0.8) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(52, 211, 153, 0.15);
    border-radius: 36px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    position: relative;
}

.header-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center, rgba(52, 211, 153, 0.1) 0%, transparent 60%);
    animation: rotate 30s linear infinite;
    pointer-events: none;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.header-box h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 5rem !important;
    font-weight: 800;
    margin-bottom: 1.5rem;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #a7f3d0 0%, #10b981 50%, #047857 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 10px 30px rgba(16, 185, 129, 0.3);
    position: relative;
    z-index: 1;
}

.header-box p {
    color: #e2e8f0;
    font-size: 1.35rem;
    font-weight: 300;
    max-width: 850px;
    margin: 0 auto;
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

/* Pulse Animation for Pill */
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.6rem 2rem;
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 2.5rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    animation: pulseGlow 3s infinite;
    position: relative;
    z-index: 1;
    backdrop-filter: blur(12px);
}

/* Floating Primary Buttons with Shine */
button.primary {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 1px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 18px !important;
    padding: 1rem 2rem !important;
    box-shadow: 0 15px 35px -5px rgba(16, 185, 129, 0.5), inset 0 2px 0 rgba(255, 255, 255, 0.3) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    position: relative;
    overflow: hidden;
}

/* Shine Hover Effect */
button.primary::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(to right, rgba(255,255,255,0) 0%, rgba(255,255,255,0.3) 50%, rgba(255,255,255,0) 100%);
    transform: rotate(45deg) translateY(-100%);
    transition: transform 0.6s ease;
}

button.primary:hover::after {
    transform: rotate(45deg) translateY(100%);
}

button.primary:hover {
    transform: translateY(-4px) scale(1.03) !important;
    box-shadow: 0 25px 45px -5px rgba(16, 185, 129, 0.6), inset 0 2px 0 rgba(255, 255, 255, 0.3) !important;
    background: linear-gradient(135deg, #34d399 0%, #10b981 100%) !important;
}

button.primary:active {
    transform: translateY(2px) scale(0.97) !important;
    box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.5) !important;
}

/* Stunning Inputs / Textareas / Dropdowns */
textarea, input[type="text"], .gr-input, .gr-box {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #f8fafc !important;
    border-radius: 16px !important;
    transition: all 0.4s ease !important;
    font-family: inherit !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
}

textarea:focus, input[type="text"]:focus, .gr-input:focus {
    border-color: #34d399 !important;
    box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.2), inset 0 2px 4px rgba(0,0,0,0.1) !important;
    background: rgba(30, 41, 59, 0.9) !important;
    outline: none !important;
    transform: translateY(-2px);
}

textarea:hover, input[type="text"]:hover, .gr-input:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Tabs Redesign */
.tabs {
    background: transparent !important;
    border: none !important;
}

.tab-nav {
    border-bottom: 2px solid rgba(255, 255, 255, 0.05) !important;
    margin-bottom: 30px !important;
    display: flex !important;
    justify-content: center !important;
    gap: 15px !important;
    padding-bottom: 15px !important;
    flex-wrap: wrap !important;
}

.tabitem {
    border: none !important;
    padding: 1.5rem 0 !important;
    background: transparent !important;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.tab-nav button {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 50px !important;
    padding: 0.8rem 1.8rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    white-space: nowrap !important;
}

.tab-nav button:hover {
    background: rgba(255, 255, 255, 0.06) !important;
    color: #f8fafc !important;
    transform: translateY(-2px);
}

.tab-nav button.selected {
    color: #064e3b !important;
    background: #34d399 !important;
    border-color: #34d399 !important;
    box-shadow: 0 10px 25px rgba(52, 211, 153, 0.4) !important;
    transform: translateY(-3px) scale(1.05);
}

/* Typography Enhancements */
label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    margin-bottom: 0.6rem !important;
    display: block !important;
    letter-spacing: 0.5px !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(52, 211, 153, 0.5);
}

/* Base Gradio Typography */
.prose {
    color: #e2e8f0 !important;
    line-height: 1.7 !important;
}
.prose strong {
    color: #6ee7b7 !important;
}
.dark {
    --body-text-color: #e2e8f0;
}
"""

theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="rgba(30, 41, 59, 0.45)",
    block_background_fill_dark="rgba(30, 41, 59, 0.45)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.05)",
    block_border_color_dark="rgba(255, 255, 255, 0.05)",
    block_radius="*radius_xl",
    block_shadow="0 10px 30px -10px rgba(0, 0, 0, 0.3)",
    button_primary_background_fill="linear-gradient(135deg, #10b981 0%, #047857 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #34d399 0%, #10b981 100%)",
    button_primary_text_color="white",
    input_background_fill="rgba(15, 23, 42, 0.6)",
    input_background_fill_dark="rgba(15, 23, 42, 0.6)",
    input_border_color="rgba(255, 255, 255, 0.08)",
    input_border_color_dark="rgba(255, 255, 255, 0.08)",
    input_border_color_focus="#10b981",
    input_border_color_focus_dark="#10b981",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#94a3b8",
)

# Gradio Interface for Bharat's Ayurvedic SLM
with gr.Blocks() as demo:
    gr.HTML("""
    <div class="header-box">
        <div class="pill">🌟 Authentic Knowledge Base</div>
        <h1>AyurSLM</h1>
        <p>Your personalized, AI-driven guide to traditional Ayurvedic health, Dosha balancing, and natural well-being.</p>
    </div>
    """)
    
    global_language = gr.Dropdown(
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
            
            with gr.Row():
                demo_btn = gr.Button("🧑‍🦱 Quick Fill Demo Profile", size="sm")
                clear_btn = gr.Button("♻️ Clear Profile", size="sm")
            
            master_btn = gr.Button("✨ Generate Complete Master Plan", variant="primary", size="lg")
            
    def load_demo():
        return "I suffer from chronic back pain, stiff joints in the morning, and very low energy.", 45, "Beginner", "Kerala, India", "Monsoon", "Vegetarian", "Irregular", "Light/Broken", "Software Engineer"
        
    def clear_profile():
        return "", 30, "Beginner", "", "Summer", "Vegetarian", "Irregular", "Light/Broken", ""

    demo_btn.click(load_demo, inputs=[], outputs=[master_symptoms, master_age, master_fitness, master_loc, master_season, master_diet, master_digestion, master_sleep, master_occ])
    clear_btn.click(clear_profile, inputs=[], outputs=[master_symptoms, master_age, master_fitness, master_loc, master_season, master_diet, master_digestion, master_sleep, master_occ])

    with gr.Tabs():
        with gr.TabItem("🩺 Symptom & Dosha Analyzer"):
            m_head_1 = gr.Markdown("### 1. Dosha Analysis")
            master_out_1 = gr.Markdown("*(Your customized dosha analysis will securely render here...)*", elem_classes=["prose"])

            
        with gr.TabItem("🌿 Herb & Remedy Finder"):
            m_head_2 = gr.Markdown("### 2. Herbal Remedies")
            master_out_2 = gr.Markdown("*(Your custom herbal remedies will securely render here...)*", elem_classes=["prose"])

            
        with gr.TabItem("🌦️ Seasonal Wellness (Ritucharya)"):
            m_head_3 = gr.Markdown("### 3. Dietary Guidelines")
            master_out_3 = gr.Markdown("*(Your personalized dietary structure will securely render here...)*", elem_classes=["prose"])

            
        with gr.TabItem("🌅 Dinacharya Planner"):
            m_head_4 = gr.Markdown("### 4. Daily Routine (Dinacharya)")
            master_out_4 = gr.Markdown("*(Your custom daily routine will securely render here...)*", elem_classes=["prose"])

            
        with gr.TabItem("🧘 Yoga & Pranayama Advisor"):
            m_head_5 = gr.Markdown("### 5. Yoga & Breathwork")
            master_out_5 = gr.Markdown("*(Your individualized yoga sequence will securely render here...)*", elem_classes=["prose"])


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


        with gr.TabItem("📖 Offline AI Agent"):
            with gr.Row():
                with gr.Column(scale=1):
                    qa_desc_md = gr.Markdown("Ask the AI agent any question. The agent will autonomously extract data from its offline resources and answer strictly based on them.")
                    qa_question = gr.Textbox(lines=2, label="Your Question", placeholder="What do the offline documents say about...")
                    qa_btn = gr.Button("Ask Offline Agent", variant="primary")
                with gr.Column(scale=1):
                    qa_output = gr.Textbox(lines=10, label="Agent's Answer", interactive=False)
                    
            qa_btn.click(ask_offline_agent, inputs=[qa_question, global_language], outputs=[qa_output])


        with gr.TabItem("💊 Drug Side Effects & Ayurveda"):
            with gr.Row():
                with gr.Column(scale=1):
                    drug_desc_md = gr.Markdown("Enter the name of a modern medicine to understand its side effects, its impact on your Doshas, and Ayurvedic ways to support your body while taking it.")
                    drug_name_in = gr.Textbox(lines=2, label="Medicine / Drug Name", placeholder="E.g., Paracetamol, Omeprazole, Ibuprofen")
                    drug_btn_in = gr.Button("Analyze Medicine", variant="primary")
                with gr.Column(scale=1):
                    drug_output_in = gr.Textbox(lines=14, label="Side Effects & Ayurvedic Analysis", interactive=False)
                    
            drug_btn_in.click(generate_drug_analysis, inputs=[drug_name_in, global_language], outputs=[drug_output_in])


    def start_loading():
        return (
            "⏳ *Analyzing your Dosha profile securely... (This may take up to a minute)*",
            "⏳ *Formulating custom herbal remedies...*",
            "⏳ *Structuring personalized dietary guidelines...*",
            "⏳ *Creating your tailored Dinacharya daily routine...*",
            "⏳ *Designing an individualized Yoga sequence...*"
        )

    master_btn.click(
        start_loading,
        inputs=[],
        outputs=[master_out_1, master_out_2, master_out_3, master_out_4, master_out_5]
    ).then(
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
            
            gr.update(value=t.get("qa_desc", "Ask the AI agent any question. The agent will autonomously extract data from its offline resources and answer strictly based on them.")),
            gr.update(label=t.get("qa_question_lbl", "Your Question"), placeholder=t.get("qa_question_pl", "What do the offline documents say about...")),
            gr.update(value=t.get("qa_btn", "Ask Offline Agent")),
            gr.update(label=t.get("qa_out", "Agent's Answer")),
            
            gr.update(value=t.get("drug_desc")),
            gr.update(label=t.get("drug_name_lbl")),
            gr.update(value=t.get("drug_btn")),
            gr.update(label=t.get("drug_out")),
            
            gr.update(value=t.get("master_desc")),
            gr.update(label=t.get("master_sym_lbl")),
            gr.update(value=t.get("master_btn")),
            gr.update(value="### " + t.get("m_out_1", "1. Dosha Analysis")),
            gr.update(value="### " + t.get("m_out_2", "2. Herbal Remedies")),
            gr.update(value="### " + t.get("m_out_3", "3. Dietary Guidelines")),
            gr.update(value="### " + t.get("m_out_4", "4. Daily Routine (Dinacharya)")),
            gr.update(value="### " + t.get("m_out_5", "5. Yoga & Breathwork"))
        ]
        
    global_language.change(
        update_ui_language,
        inputs=[global_language],
        outputs=[
            prakriti_desc_md, body_type, digestion_type, sleep_pattern, weather_pref, prakriti_btn, prakriti_output,
            qa_desc_md, qa_question, qa_btn, qa_output,
            drug_desc_md, drug_name_in, drug_btn_in, drug_output_in,
            master_desc_md, master_symptoms, master_btn, m_head_1, m_head_2, m_head_3, m_head_4, m_head_5
        ]
    )

if __name__ == "__main__":
    print("Launching AyurSLM Interface...")
    demo.launch(theme=theme, css=custom_css)
