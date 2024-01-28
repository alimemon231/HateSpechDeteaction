# calculator_api/serializers.py
from rest_framework import serializers

class SpeachSerializer(serializers.Serializer):
    
    text = serializers.CharField()
    auth = serializers.CharField()

