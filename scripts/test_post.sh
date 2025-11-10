#!/bin/bash

# POST
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19710825", "user": {"id": "aaa"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19720925", "user": {"id": "bbb"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19731025", "user": {"id": "ccc"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19740125", "user": {"id": "ddd"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19750225", "user": {"id": "eee"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19761125", "user": {"id": "fff"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
curl -k -X 'POST' 'https://0.0.0.0:8000/saju/kk_chat' -H 'accept: application/json' -H 'Content-Type: application/json' \
     -d '{"userRequest": {"utterance": "19771225", "user": {"id": "ggg"}, "callbackUrl": "https://0.0.0.0:8000/saju/test_request"}}' &
