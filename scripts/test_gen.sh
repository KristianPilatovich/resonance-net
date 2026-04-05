#!/bin/bash
# Test generation quality with various prompts

MODEL=${1:-"../checkpoints/step_0005000.bin"}
BIN="../build/resonance_net"

echo "══════════════════════════════════════════════"
echo "  ResonanceNet V5 Generation Test"
echo "  Checkpoint: $MODEL"
echo "══════════════════════════════════════════════"

echo ""
echo "─── Test 1: Story continuation ───"
$BIN infer "$MODEL" "Once upon a time"

echo ""
echo "─── Test 2: Code completion ───"
$BIN infer "$MODEL" "def fibonacci(n):"

echo ""
echo "─── Test 3: Q&A ───"
$BIN infer "$MODEL" "Question: What is 2 + 2?"

echo ""
echo "─── Test 4: Shakespeare ───"
$BIN infer "$MODEL" "To be or not to be"

echo ""
echo "─── Test 5: Pattern ───"
$BIN infer "$MODEL" "abcdefghij"

echo ""
echo "══════════════════════════════════════════════"
echo "  Done."
echo "══════════════════════════════════════════════"
