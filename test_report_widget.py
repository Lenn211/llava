#!/usr/bin/env python3
"""Test script to verify report widget updates"""

import tkinter as tk
import threading
import time

class TestReportGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Report Widget Test")
        self.root.geometry("500x400")
        
        # Create report text widget
        text_frame = tk.Frame(root, bg='#ecf0f1', relief=tk.RAISED, bd=2)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            text_frame,
            font=('Arial', 10),
            bg='#ecf0f1',
            fg='#2c3e50',
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        scrollbar.config(command=self.result_text.yview)
        
        # Add initial message
        self.result_text.insert(tk.END, "Report will appear here...\n")
        print("[INIT] Report widget created")
        print(f"[INIT] Initial content: {repr(self.result_text.get('1.0', tk.END))}")
        
        # Create test buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Test Direct Insert", 
                  command=self.test_direct).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Test add_result()", 
                  command=self.test_add_result).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Test from Thread", 
                  command=self.test_from_thread).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="Clear", 
                  command=lambda: self.result_text.delete("1.0", tk.END)).pack(side=tk.LEFT, padx=5)
    
    def test_direct(self):
        """Test direct insert"""
        print("\n[TEST 1] Direct insert")
        self.result_text.insert(tk.END, "Direct insert test\n")
        self.result_text.see(tk.END)
        print("[TEST 1] Complete")
    
    def add_result(self, message, tag='normal'):
        """Mimic the add_result function from the main GUI"""
        def _add():
            try:
                print(f"[add_result] _add() executing: {message}")
                self.result_text.insert(tk.END, message + "\n")
                self.result_text.see(tk.END)
                self.result_text.update_idletasks()
                print(f"[add_result] Success")
            except Exception as e:
                print(f"[add_result] ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[add_result] Scheduling via root.after()")
        self.root.after(0, _add)
    
    def test_add_result(self):
        """Test the add_result function"""
        print("\n[TEST 2] Testing add_result()")
        self.add_result("Test message via add_result()")
    
    def test_from_thread(self):
        """Test adding from a background thread"""
        print("\n[TEST 3] Testing from background thread")
        
        def _thread_work():
            print("[Thread] Starting")
            time.sleep(1)
            print("[Thread] Calling add_result()")
            self.add_result("Message from background thread!")
            print("[Thread] Done")
        
        thread = threading.Thread(target=_thread_work, daemon=True)
        thread.start()
        print("[TEST 3] Thread started")

if __name__ == "__main__":
    root = tk.Tk()
    app = TestReportGUI(root)
    print("\n" + "="*50)
    print("GUI started. Click the buttons to test.")
    print("Watch the terminal for debug output.")
    print("="*50 + "\n")
    root.mainloop()
