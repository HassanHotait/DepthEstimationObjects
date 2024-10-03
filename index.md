---
layout: default
title: Rerun Web Viewer
---

# Rerun Web Viewer Integration

This page displays a live Rerun web viewer that is running on a server.

<!-- Embed the Rerun viewer using an iframe -->
<iframe src="192.168.56.1:9876" width="100%" height="600" frameborder="0" allowfullscreen></iframe>

> Make sure the Rerun server is running and accessible at the provided URL.

## How to Use

To visualize your data:
1. Start the Rerun server using the following command:
   ```bash
   rerun --serve
