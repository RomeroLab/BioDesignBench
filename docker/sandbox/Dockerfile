FROM python:3.10-slim

# Install common bioinformatics dependencies
RUN pip install --no-cache-dir \
    biopython \
    requests \
    numpy \
    pandas \
    scipy \
    pyyaml

WORKDIR /workspace

# Agent-generated code runs here
CMD ["python", "script.py"]
