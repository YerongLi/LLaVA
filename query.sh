# curl -X POST -H "Content-Type: application/json" -d '{"text": "What is this?", "model":"llava-v1.5-13b"}' http://localhost:10000/worker_generate_stream --output "result.txt"
# --model-path "/scratch/yerong/.cache/pyllama/llava-v1.5-13b/" \ 
python -m query   \
  --model-path "/scratch/yerong/.cache/pyllama/llava-v1.5-13b/"