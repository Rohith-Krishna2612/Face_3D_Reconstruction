import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styled from 'styled-components';

const UploadContainer = styled.div`
  margin: 40px 0;
`;

const DropzoneArea = styled.div`
  border: 3px dashed ${props => props.isDragActive ? '#28a745' : '#007bff'};
  border-radius: 15px;
  padding: 60px 30px;
  text-align: center;
  background: ${props => props.isDragActive ? '#e6ffe6' : '#f8f9ff'};
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    border-color: #0056b3;
    background: #e6f0ff;
  }
`;

const UploadIcon = styled.div`
  font-size: 4rem;
  margin-bottom: 20px;
`;

const UploadText = styled.h3`
  font-size: 1.3rem;
  color: #333;
  margin-bottom: 10px;
`;

const UploadSubtext = styled.p`
  color: #666;
  font-size: 0.95rem;
  margin-bottom: 20px;
`;

const FileInput = styled.input`
  display: none;
`;

const UploadButton = styled.button`
  background: #007bff;
  color: white;
  border: none;
  padding: 12px 30px;
  border-radius: 25px;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: #0056b3;
    transform: translateY(-2px);
  }
`;

const ErrorMessage = styled.div`
  color: #dc3545;
  background: #f8d7da;
  border: 1px solid #f5c6cb;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
  text-align: center;
`;

const PreviewContainer = styled.div`
  margin-top: 30px;
  text-align: center;
`;

const PreviewImage = styled.img`
  max-width: 300px;
  max-height: 300px;
  border-radius: 10px;
  box-shadow: 0 5px 15px rgba(0,0,0,0.2);
`;

const ImageInfo = styled.div`
  margin-top: 15px;
  color: #666;
  font-size: 0.9rem;
`;

function ImageUploader({ onImageUpload, error }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.webp']
    },
    multiple: false,
    maxSize: 10 * 1024 * 1024 // 10MB
  });

  const handleUpload = () => {
    if (selectedFile) {
      onImageUpload(selectedFile);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreview(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <UploadContainer>
      <DropzoneArea {...getRootProps()} isDragActive={isDragActive}>
        <input {...getInputProps()} />
        <UploadIcon>
          {isDragActive ? '📤' : '📷'}
        </UploadIcon>
        <UploadText>
          {isDragActive
            ? 'Drop the image here!'
            : 'Upload Face Image'
          }
        </UploadText>
        <UploadSubtext>
          Drag and drop an image here, or click to browse<br/>
          Supports: JPEG, PNG, BMP, WebP (Max 10MB)
        </UploadSubtext>
      </DropzoneArea>

      {selectedFile && (
        <PreviewContainer>
          <h3>Selected Image Preview</h3>
          <PreviewImage src={preview} alt="Preview" />
          <ImageInfo>
            <strong>{selectedFile.name}</strong><br/>
            Size: {formatFileSize(selectedFile.size)}<br/>
            Type: {selectedFile.type}
          </ImageInfo>
          <div style={{ marginTop: '20px' }}>
            <UploadButton onClick={handleUpload}>
              🚀 Process Image
            </UploadButton>
            <UploadButton 
              onClick={handleReset} 
              style={{ marginLeft: '15px', background: '#6c757d' }}
            >
              🔄 Choose Different Image
            </UploadButton>
          </div>
        </PreviewContainer>
      )}

      {error && (
        <ErrorMessage>
          <strong>Error:</strong> {error}
        </ErrorMessage>
      )}
    </UploadContainer>
  );
}

export default ImageUploader;