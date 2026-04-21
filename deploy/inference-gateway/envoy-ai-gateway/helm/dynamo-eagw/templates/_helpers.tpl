# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{/*
Expand the name of the chart.
*/}}
{{- define "dynamo-eagw.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name (≤63 chars, DNS-safe).
*/}}
{{- define "dynamo-eagw.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name + version label value.
*/}}
{{- define "dynamo-eagw.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "dynamo-eagw.labels" -}}
helm.sh/chart: {{ include "dynamo-eagw.chart" . }}
{{ include "dynamo-eagw.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "dynamo-eagw.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dynamo-eagw.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Resolve the Dynamo namespace as "<release-namespace>-<dynamoGraphDeploymentName>".
This matches how the Dynamo operator names its per-deployment namespaces.
*/}}
{{- define "dynamo-eagw.dynamoNamespace" -}}
{{- $dgdName := (.Values.dynamoGraphDeploymentName | default "") | trim -}}
{{- if not $dgdName }}
{{- fail "dynamoGraphDeploymentName must be set" }}
{{- end }}
{{- $releaseNs := (.Release.Namespace | default "") | trim -}}
{{- if not $releaseNs }}
{{- fail "Release.Namespace must be set" }}
{{- end }}
{{- printf "%s-%s" $releaseNs $dgdName }}
{{- end }}

{{/*
Resolve the Gateway namespace (defaults to release namespace).
*/}}
{{- define "dynamo-eagw.gatewayNamespace" -}}
{{- default .Release.Namespace .Values.aiGatewayRoute.gatewayNamespace }}
{{- end }}
