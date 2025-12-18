#!/usr/bin/env python3
"""Minimal test to verify Text widget updates from threads"""

import tkinter as tk
import threading
import time

def test_direct_insert(text_widget):
    """Test direct insert (NOT thread-safe)"""
    print("[TEST 1] Direct insert")
    text_widget.insert(tk.END, "Direct insert test\n")
    print("[TEST 1] Complete")

def test_after_insert(root, text_widget):
    """Test insert via root.after() (thread-safe)"""
    print("[TEST 2] After insert")
    def _insert():
        print("[TEST 2] _insert() executing")
        text_widget.insert(tk.END, "After insert test\n")
        print("[TEST 2] Insert complete")
    root.after(0, _insert)
    print("[TEST 2] Scheduled")

def test_thread_direct(text_widget):
    """Test direct insert from thread (BAD - not thread-safe)"""
    def _thread_work():
        print("[TEST 3] Thread direct - starting")
        time.sleep(0.5)
        print("[TEST 3] Thread direct - inserting")
        text_widget.insert(tk.END, "Thread direct insert (BAD!)\n")
        print("[TEST 3] Thread direct - complete")
    
    thread = threading.Thread(target=_thread_work, daemon=True)
    thread.start()

def test_thread_after(root, text_widget):
    """Test insert from thread via root.after() (GOOD - thread-safe)"""
    def _thread_work():
        print("[TEST 4] Thread after - starting")
        time.sleep(0.5)
        print("[TEST 4] Thread after - scheduling insert")
        def _insert():
            print("[TEST 4] _insert() executing")
            text_widget.insert(tk.END, "Thread after insert (GOOD!)\n")
            print("[TEST 4] Insert complete")
        root.after(0, _insert)
        print("[TEST 4] Thread after - scheduled")
    
    thread = threading.Thread(target=_thread_work, daemon=True)
    thread.start()

def main():
    root = tk.Tk()
    root.title("Text Widget Thread Test")
    root.geometry("600x400")
    
    # Create text widget
    frame = tk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    text = tk.Text(frame, yscrollcommand=scrollbar.set, wrap=tk.WORD)
    text.pack(fill=tk.BOTH, expand=True)
    scrollbar.config(command=text.yview)
    
    text.insert(tk.END, "=== Text Widget Thread Safety Test ===\n\n")
    
    # Create test buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    
    tk.Button(btn_frame, text="Test 1: Direct", 
              command=lambda: test_direct_insert(text)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(btn_frame, text="Test 2: After", 
              command=lambda: test_after_insert(root, text)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(btn_frame, text="Test 3: Thread Direct (BAD)", 
              command=lambda: test_thread_direct(text)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(btn_frame, text="Test 4: Thread After (GOOD)", 
              command=lambda: test_thread_after(root, text)).pack(side=tk.LEFT, padx=5)
    
    tk.Button(btn_frame, text="Clear", 
              command=lambda: text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
