import wandb
# from transformers import pipeline

# Step 1: Initialize WandB
wandb.init(project="openvla", entity="openvla_cvpr", name="single-file-example")

# Step 2: Load a text generation model (GPT-2 in this case)
# generator = pipeline("text-generation", model="gpt2")

# Step 3: Define input prompts
prompts = [
    "Once upon a time",
    "The future of AI is",
    "In a galaxy far, far away"
]

# Step 4: Generate outputs and log them to WandB
for prompt in prompts:
    # Generate text from the LLM
    # output = generator(prompt, max_length=50, num_return_sequences=1)
    # randomly generate some output
    output = [{"generated_text": "This is a randomly generated output."}]
    generated_text = output[0]['generated_text']

    # Log the input and output to WandB
    wandb.log({"Input Prompt": prompt, "Generated Text": generated_text})   # ???
    print(f"Logged to WandB: {prompt} -> {generated_text}")

# Step 5: Finish the WandB run
wandb.finish()

print("All prompts and outputs have been logged to WandB!")

# python scripts/test_wandb.py