# Use the prebuilt base image
FROM audiobook-tts-base  

WORKDIR /app

# Set Hugging Face cache directory so large models are cached here
ENV HF_HOME=/app/hf_cache

# Now copy the rest of the source files (to avoid cache invalidation on requirements installation)
COPY audiobook-tts/run.sh ./


# Ensure the runtime script is executable
RUN chmod +x run.sh

# Create necessary directories
RUN mkdir -p temp 3-output hf_cache 

# Set the default command
CMD ["./run.sh"]
