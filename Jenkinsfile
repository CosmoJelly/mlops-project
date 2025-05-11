pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/CosmoJelly/mlops-project.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t mlops-project:latest .'
            }
        }
        stage('Deploy to Minikube') {
            steps {
                sh 'minikube image load mlops-project:latest'
                sh 'kubectl apply -f deployment.yaml'
                sh 'kubectl apply -f service.yaml'
            }
        }
    }
}