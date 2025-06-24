import vnstock
import inspect
import json
from pprint import pprint

def analyze_module(module):
    """Phân tích cấu trúc của module"""
    structure = {
        'functions': [],
        'classes': [],
        'variables': []
    }
    
    # Lấy tất cả các thuộc tính của module
    for name, obj in inspect.getmembers(module):
        # Bỏ qua các thuộc tính private và special
        if name.startswith('_'):
            continue
            
        # Phân loại các đối tượng
        if inspect.isfunction(obj):
            func_info = {
                'name': name,
                'doc': inspect.getdoc(obj),
                'signature': str(inspect.signature(obj))
            }
            structure['functions'].append(func_info)
            
        elif inspect.isclass(obj):
            class_info = {
                'name': name,
                'doc': inspect.getdoc(obj),
                'methods': []
            }
            # Lấy các method của class
            for method_name, method_obj in inspect.getmembers(obj, predicate=inspect.isfunction):
                if not method_name.startswith('_'):
                    method_info = {
                        'name': method_name,
                        'doc': inspect.getdoc(method_obj),
                        'signature': str(inspect.signature(method_obj))
                    }
                    class_info['methods'].append(method_info)
            structure['classes'].append(class_info)
            
        else:
            # Các biến và hằng số
            structure['variables'].append({
                'name': name,
                'type': str(type(obj))
            })
    
    return structure

def print_structure(structure):
    """In cấu trúc phân tích được theo định dạng dễ đọc"""
    print("\n=== FUNCTIONS ===")
    for func in structure['functions']:
        print(f"\n📌 {func['name']}{func['signature']}")
        if func['doc']:
            print(f"   📝 {func['doc']}")
            
    print("\n=== CLASSES ===")
    for class_ in structure['classes']:
        print(f"\n🔷 {class_['name']}")
        if class_['doc']:
            print(f"   📝 {class_['doc']}")
        print("   Methods:")
        for method in class_['methods']:
            print(f"   ├─ {method['name']}{method['signature']}")
            if method['doc']:
                print(f"   │  📝 {method['doc']}")
                
    print("\n=== VARIABLES ===")
    for var in structure['variables']:
        print(f"• {var['name']}: {var['type']}")

def main():
    print("🔍 Analyzing vnstock library structure...")
    structure = analyze_module(vnstock)
    
    print("\n📊 VNSTOCK LIBRARY STRUCTURE")
    print("=" * 50)
    print_structure(structure)
    
    # Lưu kết quả vào file JSON để tham khảo sau
    with open('vnstock_structure.json', 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=2, ensure_ascii=False)
    print("\n✅ Analysis complete! Results saved to vnstock_structure.json")

if __name__ == "__main__":
    main() 