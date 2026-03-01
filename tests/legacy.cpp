#include <iostream>
#include <vector>

namespace LegacySystem {
    
    // Test: Class Extraction
    class DataProcessor {
    public:
        // Test: Nested Function Extraction
        void initialize(int speed) {
            std::cout << "Processor ready at " << speed << std::endl;
        }

        // Test: Virtual/Abstract functions
        virtual bool validateData(const std::string& raw) = 0;
    };

    // Test: Template Class
    template <typename T>
    class MemoryBuffer : public DataProcessor {
    private:
        std::vector<T> buffer;
    public:
        void push(T item) {
            buffer.push_back(item);
        }
        
        // Test: Function with complex return type
        std::vector<T>& getInternalBuffer() {
            return buffer;
        }

        bool validateData(const std::string& raw) override {
            return !raw.empty();
        }
    };
}

// Test: Global Function
int main() {
    LegacySystem::MemoryBuffer<int> mem;
    mem.initialize(100);
    return 0;
}