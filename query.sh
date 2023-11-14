# curl -X POST -H "Content-Type: application/json" -d '{"text": "What is this?", "model":"llava-v1.5-13b"}' http://localhost:10000/worker_generate_stream --output "result.txt"
python -m query   \
  --model-path "/scratch/yerong/.cache/pyllama/llava-v1.5-13b/" \
   --image-file "https://llava-vl.github.io/static/images/view.jpg"
