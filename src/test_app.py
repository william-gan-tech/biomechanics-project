from streamlit.testing.v1 import AppTest

def test_app_initialization():
    # Initialize the app from your main dashboard script file
    at = AppTest.from_file("dashboard.py").run()
    
    # Assert that the app ran without uncaught exceptions
    assert not at.exception
    
    # Verify that headers or primary titles load correctly
    assert len(at.header) > 0

def test_mode_5_pipeline_workflow():
    at = AppTest.from_file("dashboard.py").run()
    
    # Example: If you want to simulate selecting a radio option or clicking execution buttons
    # Locate elements by their index or key and trigger interactions
    if len(at.button) > 0:
        # Simulate clicking the first available execution/test button
        at.button[0].click().run()
        
        # Verify state or check that metrics containers updated successfully
        assert not at.exception