import sys
import pkg_resources
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

def check_installation():
    """Check if Ragas and its dependencies are installed correctly"""
    print("Checking Ragas installation...")
    
    # Check Ragas version
    try:
        ragas_version = pkg_resources.get_distribution("ragas").version
        print(f"✓ Ragas version {ragas_version} is installed")
    except pkg_resources.DistributionNotFound:
        print("✗ Ragas is not installed")
        return False
    
    # Check other required packages
    required_packages = {
        "datasets": "2.12.0",
        "torch": "2.0.0",
        "transformers": "4.30.0",
        "sentence-transformers": "2.2.2"
    }
    
    all_installed = True
    for package, min_version in required_packages.items():
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✓ {package} version {version} is installed")
        except pkg_resources.DistributionNotFound:
            print(f"✗ {package} is not installed")
            all_installed = False
    
    return all_installed

def test_basic_functionality():
    """Test basic Ragas functionality with a simple example"""
    print("\nTesting basic Ragas functionality...")
    
    # Create a simple test dataset
    test_data = {
        "question": ["What is the capital of France?"],
        "ground_truth": ["The capital of France is Paris."],
        "contexts": [["France is a country in Europe. Its capital is Paris."]],
        "answer": ["Paris is the capital of France."]
    }
    
    try:
        # Create dataset
        dataset = Dataset.from_dict(test_data)
        
        # Run a simple evaluation
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy]
        )
        
        print("✓ Basic evaluation completed successfully")
        print("\nTest Results:")
        for metric, score in result.items():
            print(f"{metric}: {score:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Error during basic functionality test: {str(e)}")
        return False

def main():
    print("=== Ragas Installation and Functionality Test ===\n")
    
    # Check installation
    if not check_installation():
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\nBasic functionality test failed. Please check your installation.")
        sys.exit(1)
    
    print("\n=== All tests passed successfully! ===")
    print("Ragas is installed correctly and working as expected.")

if __name__ == "__main__":
    main() 